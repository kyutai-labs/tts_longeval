# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"A turn based version of moshi. Much slower, no batching."

import argparse
import json
from pathlib import Path
import sys

import sphn
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO
from external_tools.audio import Smoother
from external_tools.speaker import hf_get


@torch.no_grad()
def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-repo", type=str, default=DEFAULT_DSM_TTS_REPO, help="HF repo in which to look for the pretrained models."
    )
    parser.add_argument("--config", "--lm-config", dest="config", type=str, help="The config as a json file.")

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to be used for inference.")
    parser.add_argument("--nq", type=int, default=32, help="Number of codebooks to generate.")
    parser.add_argument("--temp", type=float, default=0.6, help="Temperature for text and audio.")
    parser.add_argument("--cfg-coef", type=float, default=2.0, help="CFG coefficient.")

    parser.add_argument(
        "--cfg-has-text",
        action="store_false",
        dest="cfg_is_no_text",
        default=True,
        help="Disable the fact that the CFG has no text.",
    )
    parser.add_argument(
        "--cfg-has-prefix",
        action="store_false",
        dest="cfg_is_no_prefix",
        default=True,
        help="Disable the fact that the CFG has no prefix.",
    )

    parser.add_argument("--max-padding", type=int, default=8, help="Max padding in a row.")
    parser.add_argument("--initial-padding", type=int, default=2, help="Initial padding.")
    parser.add_argument("--final-padding", type=int, default=4, help="Final padding.")
    parser.add_argument("--padding-bonus", type=float, default=0.0, help="Bonus to the padding logits.")
    parser.add_argument(
        "--padding-between", type=int, default=1, help="Forces a minimal amount of fixed padding between words."
    )

    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )

    args = parser.parse_args()

    checkpoint_info = CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config
    )

    cfg_coef_conditioning = None
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        n_q=args.nq,
        temp=args.temp,
        cfg_coef=args.cfg_coef,
        max_padding=args.max_padding,
        initial_padding=args.initial_padding,
        final_padding=args.final_padding,
        padding_bonus=args.padding_bonus,
        device=args.device,
        dtype=args.dtype,
    )
    if tts_model.valid_cfg_conditionings:
        # Model was trained with CFG distillation.
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = args.cfg_is_no_text
        cfg_is_no_prefix = args.cfg_is_no_prefix
    mimi = tts_model.mimi

    while True:
        line = sys.stdin.readline()
        items = json.loads(line)
        assert len(items) == 1
        item = items[0]

        segments = []
        start = 0.0
        all_wavs = []
        for idx, turn in enumerate(item["turns"]):
            speaker_idx = idx % len(item["speaker_audios"])
            speaker = item["speaker_audios"][speaker_idx]
            print("Generating", turn)
            entries = tts_model.prepare_script([turn], padding_between=args.padding_between)
            if tts_model.multi_speaker:
                voices = [hf_get(speaker + tts_model.voice_suffix)]
            else:
                voices = []
            attributes = tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
            prefixes = None
            if not tts_model.multi_speaker:
                prefixes = [tts_model.get_prefix(hf_get(speaker))]

            result = tts_model.generate(
                [entries],
                [attributes],
                prefixes=prefixes,
                cfg_is_no_prefix=cfg_is_no_prefix,
                cfg_is_no_text=cfg_is_no_text,
            )
            frames = torch.cat(result.frames, dim=-1).cpu()

            wav_frames = []
            with torch.no_grad(), mimi.streaming(len(frames)):
                for frame in result.frames[tts_model.delay_steps :]:
                    wav_frames.append(mimi.decode(frame[:, 1:]))
            wavs = torch.cat(wav_frames, dim=-1)
            idx = 0
            end_step = result.end_steps[idx]
            if end_step is None:
                print(f"Warning: end step is None, generation failed for {item['output_file']}")
                wav_length = wavs.shape[-1]
            else:
                wav_length = int((mimi.sample_rate * (end_step + tts_model.final_padding) / mimi.frame_rate))
            wav = wavs[idx, :, :wav_length]
            start_time = 0.0
            if prefixes is not None:
                start_time = prefixes[idx].shape[-1] / mimi.frame_rate
            start_sample = int(start_time * mimi.sample_rate)
            wav = wav[:, start_sample:]
            duration = wav.shape[-1] / mimi.sample_rate
            wav = Smoother()(wav.cpu())
            all_wavs.append(wav)
            segments.append((speaker_idx, (start, start + duration)))
            start += duration

        wav = torch.cat(all_wavs, dim=-1)
        filename = Path(item["output_file"])
        sphn.write_wav(filename, wav.clamp(-0.99, 0.99).cpu().numpy(), mimi.sample_rate)
        with open(filename.with_suffix(".segments.json"), "w") as f:
            json.dump({"segments": segments}, f)
        print("Saved", filename)

        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
