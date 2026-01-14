# Copyright (c) Kyutai, all rights reserved.

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

from safetensors.torch import save_file
import sphn
import torch

from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_REPO
from external_tools.audio import audio_write
from external_tools.speaker import hf_get


@dataclass
class TTSRequest:
    """Format for a single TTS request in the provided JSONL file.

    Args:
        script: list of strings, one per turn, starting with the MAIN speaker.
            If you want to generate single turn, just put a single entry.
        voices: list of voice names, starting with MAIN. Put a single voice for single
            speaker generation (or you can try twice the same voice). There should be
            a file in your voice folder (see `--voice-folder` argument) with this name + '.safetensors'.
        extra: extra fields from the input json that are not used by the model.
    """

    turns: list[str]
    voices: list[str]
    output_file: str
    extra: dict[str, Any]


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

    parser.add_argument("--shuffle-cross", action="store_true", help="Used for debugging.")
    parser.add_argument("--no-voices", action="store_true", help="Used for debugging.")
    parser.add_argument("--voice-is-prefix", action="store_true", help="Used for debugging.")

    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Run inference with float16, not bfloat16, better for old GPUs.",
    )
    parser.add_argument("--float", action="store_const", const=torch.float32, dest="dtype")

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
        if cfg_coef_conditioning < 0:
            cfg_coef_conditioning = None
        tts_model.cfg_coef = 1
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = args.cfg_is_no_text
        cfg_is_no_prefix = args.cfg_is_no_prefix
    mimi = tts_model.mimi

    @torch.no_grad()
    def _flush():
        all_entries = []
        all_attributes = []
        prefixes = None
        if not tts_model.multi_speaker or args.voice_is_prefix:
            prefixes = []
        begin = time.time()
        for request in batch:
            entries = tts_model.prepare_script(request.turns, padding_between=args.padding_between)
            all_entries.append(entries)
            if tts_model.multi_speaker:
                voices = [hf_get(voice + tts_model.voice_suffix) for voice in request.voices]
            else:
                voices = []
            all_attributes.append(tts_model.make_condition_attributes(voices, cfg_coef_conditioning))
            if args.no_voices:
                all_attributes[-1].tensor["speaker_wavs"].mask[:] = 0
                all_attributes[-1].tensor["speaker_wavs"].tensor[:] = 0
            if prefixes is not None:
                assert len(request.voices) == 1, "For this model, at most one voice is supported."
                prefixes.append(tts_model.get_prefix(hf_get(request.voices[0])))

        if args.shuffle_cross:
            perm = torch.randperm(len(all_attributes)).tolist()
            all_attributes = [all_attributes[p] for p in perm]

        print(f"Starting batch of size {len(batch)}")
        result = tts_model.generate(
            all_entries,
            all_attributes,
            prefixes=prefixes,
            cfg_is_no_prefix=cfg_is_no_prefix,
            cfg_is_no_text=cfg_is_no_text,
        )
        frames = torch.cat(result.frames, dim=-1).cpu()
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        print(f"[LM] Batch of size {len(batch)} took {time_taken:.2f}s, total speed {total_speed:.2f}x")

        wav_frames = []
        with torch.no_grad(), mimi.streaming(len(all_entries) * (1 + tts_model.multistream)):
            for frame in result.frames[tts_model.delay_steps :]:
                if tts_model.multistream:
                    n_q = tts_model.n_q // 2
                    left, right = frame[:, 1:].chunk(2, dim=1)
                    stereo_frame = torch.cat([left[:, :n_q], right[:, :n_q]], dim=0)
                    left_audio, right_audio = mimi.decode(stereo_frame).chunk(2, dim=0)
                    wav_frames.append(torch.cat([left_audio, right_audio], dim=1))
                else:
                    wav_frames.append(mimi.decode(frame[:, 1:]))
        wavs = torch.cat(wav_frames, dim=-1)
        effective_duration = 0.0
        for idx, request in enumerate(batch):
            end_step = result.end_steps[idx]
            if end_step is None:
                print(f"Warning: end step is None, generation failed for {request.output_file}")
                wav_length = wavs.shape[-1]
            else:
                wav_length = int((mimi.sample_rate * (end_step + tts_model.final_padding) / mimi.frame_rate))
            effective_duration += wav_length / mimi.sample_rate
            wav = wavs[idx, :, :wav_length]
            start_time = 0.0

            if prefixes is not None:
                start_time = prefixes[idx].shape[-1] / mimi.frame_rate

            start = int(start_time * mimi.sample_rate)
            wav = wav[:, start:]
            duration = wav.shape[-1] / mimi.sample_rate
            filename = Path(request.output_file)
            debug_tensors = {
                "frames": frames[idx].int(),
            }
            if prefixes is not None:
                debug_tensors["prefix_codes"] = prefixes[idx]

            segments = []
            transcript = []
            last_segment_start = 0
            last_speaker = None
            segment_has_content = False
            for entry, step in zip(all_entries[idx], result.all_consumption_times[idx]):
                if not entry.tokens:
                    continue
                timestamp = step / mimi.frame_rate - start_time
                if entry.text:
                    segment_has_content = True
                    transcript.append((entry.text, timestamp))
                if entry.tokens:
                    speakers = [tts_model.machine.token_ids.main, tts_model.machine.token_ids.other]
                    try:
                        speaker = speakers.index(entry.tokens[0])
                    except ValueError:
                        pass
                    else:
                        if last_speaker is not None:
                            assert speaker != last_speaker, (speaker, last_speaker, timestamp, entry.text)
                            segments.append((last_speaker, (last_segment_start, timestamp)))
                            last_segment_start = timestamp
                            segment_has_content = False
                        last_speaker = speaker
            if segment_has_content:
                if last_speaker is None:
                    assert not tts_model.multi_speaker, "No speaker token, but model is supposed to be multi speaker."
                    last_speaker = 0
                segments.append((last_speaker, (last_segment_start, duration)))

            if filename.suffix == ".wav":
                sphn.write_wav(filename, wav.clamp(-0.99, 0.99).cpu().numpy(), mimi.sample_rate)
            else:
                audio_write(filename, wav, mimi.sample_rate)

            save_file(debug_tensors, filename.with_suffix(".safetensors"))
            entries = all_entries[idx]
            debug_info = {
                "hf_repo": args.hf_repo,
                "model_id": checkpoint_info.model_id,
                "cfg_coef": tts_model.cfg_coef,
                "temp": tts_model.temp,
                "max_padding": tts_model.machine.max_padding,
                "initial_padding": tts_model.machine.initial_padding,
                "final_padding": tts_model.final_padding,
                "padding_between": args.padding_between,
                "transcript": transcript,
                "segments": segments,
                "consumption_times": result.all_consumption_times[idx],
                "turns": request.turns,
                "voices": request.voices,
                "logged_text_tokens": result.logged_text_tokens[idx],
                "end_step": end_step,
                "start_time": start_time,
                **request.extra,
            }
            with open(filename.with_suffix(".json"), "w") as f:
                json.dump(debug_info, f)
            with open(filename.with_suffix(".segments.json"), "w") as f:
                json.dump({"segments": segments}, f)
            print("Saved", filename)
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        effective_speed = effective_duration / time_taken
        print(
            f"[TOT] Batch of size {len(batch)} took {time_taken:.2f}s, "
            f"total speed {total_speed:.2f}x, "
            f"effective speed {effective_speed:.2f}x"
        )
        batch.clear()

    while True:
        batch: list[TTSRequest] = []
        line = sys.stdin.readline()
        items = json.loads(line)
        for item in items:
            if len(item["speaker_audios"]) == 1:
                turns = [" ".join(turn.strip() for turn in item["turns"])]
            else:
                turns = item["turns"]
            batch.append(
                TTSRequest(
                    voices=item["speaker_audios"],
                    turns=turns,
                    extra=item["extra"],
                    output_file=item["output_file"],
                )
            )
        _flush()
        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
