# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/SesameAILabs/csm
# originally released under the Apache 2 license, a copy
# is available in LICENSE-apache.
"""Wrapper around CSM from SesameAI."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio  # type: ignore
from external_tools.speaker import Smoother, get_speaker_audio


def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(1, "./csm")
    # Disable Triton compilation
    os.environ["NO_TORCH_COMPILE"] = "1"

    from generator import Segment, load_csm_1b  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--single-mode", choices=["double", "default"], default="default")
    parser.add_argument("-c", "--context", type=int)
    parser.add_argument("-t", "--turns", type=int, default=1)
    parser.add_argument("-d", "--max-audio-duration", type=float, default=10.0)
    args = parser.parse_args()
    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    generator = load_csm_1b(device)

    while True:
        print("Ready to process batch.")
        line = sys.stdin.readline()
        batch = json.loads(line)
        assert len(batch) == 1
        item = batch[0]

        single_speaker = len(item["speaker_audios"]) == 1
        if single_speaker:
            if args.single_mode == "double":
                item["speaker_audios"] *= 2
                single_speaker = False
            else:
                assert args.single_mode == "default"

        prompt_segments = []
        for idx, speaker_audio in enumerate(item["speaker_audios"]):
            audio, text = get_speaker_audio(speaker_audio, generator.sample_rate)
            prompt = Segment(text=text, speaker=idx, audio=audio[0])
            prompt_segments.append(prompt)

        conversation = []
        for idx, turn in enumerate(item["turns"]):
            conversation.append({"text": turn, "speaker_id": idx % len(prompt_segments)})

        # Generate each utterance
        generated_segments = []
        context = []
        segments = []
        start = 0.0
        n_speakers = len(prompt_segments)
        for utterance in conversation:
            print(f"Generating: {utterance['text']}")
            trials = 5
            spk_id = utterance["speaker_id"]
            if args.context is not None:
                if args.context == 0:
                    context = []
                else:
                    context = context[-args.context :]

            for trial in range(trials):
                this_prompts = prompt_segments
                if context:
                    need_swap = context[0].speaker == 1
                else:
                    need_swap = spk_id == 1
                if need_swap:
                    this_prompts = prompt_segments[::-1]
                spks = [x.speaker for x in this_prompts + context]
                print("speaker list", spks, "next", spk_id)
                if not single_speaker:
                    assert spks[-1] != spk_id, (spks[-1], spk_id)
                    last = spks[0]
                    for spk in spks[1:]:
                        assert spk != last, spks
                        last = spk
                    assert last != spk_id, (spks, spk_id)
                this_context = []
                last = None
                for idx, seg in enumerate(this_prompts + context):
                    this_context.append(Segment(text=seg.text, speaker=idx % n_speakers, audio=seg.audio))
                new_spk_id = (this_context[-1].speaker + 1) % n_speakers
                print("new speaker list", [x.speaker for x in this_context], "next", new_spk_id)
                try:
                    audio_tensor = generator.generate(
                        text=utterance["text"],
                        speaker=new_spk_id,
                        context=this_context,
                        max_audio_length_ms=args.max_audio_duration * 1000,
                    )
                except ValueError:
                    if trial == trials - 1:
                        raise
                    print("Input probably too long, removing some context.")
                    if len(context):
                        if trial == trials - 2:
                            context = []
                        else:
                            context = context[len(prompt_segments) :]
                    else:
                        context = []
                        raise
                except RuntimeError:
                    if trial == trials - 1:
                        raise
                    print("The model output nothing, retrying for a bit.")
                else:
                    audio = Smoother()(audio_tensor.cpu())
                    duration = audio.shape[-1] / generator.sample_rate
                    segments.append((spk_id, (start, start + duration)))
                    start += duration
                    segment = Segment(text=utterance["text"], speaker=utterance["speaker_id"], audio=audio_tensor)
                    generated_segments.append(segment)
                    context.append(segment)
                    break

        # Concatenate all generations
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        all_audio.clamp_(-0.99, 0.99)
        torchaudio.save(item["output_file"], all_audio.unsqueeze(0).cpu(), generator.sample_rate)
        with open(Path(item["output_file"]).with_suffix(".segments.json"), "w") as f:
            json.dump({"segments": segments}, f)
        print(f"Successfully generated {item['output_file']}")
        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
