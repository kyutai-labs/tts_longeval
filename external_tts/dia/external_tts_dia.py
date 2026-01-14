# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"Wrapper around DIA https://github.com/nari-labs/dia."

import argparse
import json
import sys
import traceback
from collections import deque
from pathlib import Path

import sphn
import torch
from external_tools.speaker import Smoother, get_speaker_audio


def build_audio_text(
    model,
    entries: list[tuple[int, str, torch.Tensor | None]],
    audio_prompts: list[torch.Tensor],
    text_prompts: list[str],
    context_length: int,
):
    speakers = {}
    for speaker, *_ in entries[context_length:]:
        if speaker not in speakers:
            speakers[speaker] = len(speakers)
    for speaker, *_ in entries[:context_length]:
        if speaker not in speakers:
            speakers[speaker] = len(speakers)

    audios = []
    texts = []
    last = None
    for speaker, speaker_idx in speakers.items():
        text = text_prompts[speaker].strip()
        text = f"[S{speaker_idx + 1}] {text}"
        print("ADDING SPEAKER", speaker, speaker_idx, text)
        audios.append(audio_prompts[speaker])
        texts.append(text)
        last = speaker

    for speaker, text, audio in entries:
        speaker_idx = speakers[speaker]
        text = text.strip()
        if last != speaker:
            text = f"[S{speaker_idx + 1}] {text}"
        texts.append(text)
        if audio is not None:
            audios.append(audio)
        last = speaker

    audio = torch.cat(audios, dim=-1)
    from dia.model import DEFAULT_SAMPLE_RATE  # type: ignore

    sphn.write_wav("/home/alex/tmp/dbg_context.wav", audio.numpy(), DEFAULT_SAMPLE_RATE)
    audio_tokens = model._encode(audio.to(model.device))
    text = " ".join(texts)
    print("GENERATING", text)
    return text, audio_tokens


def main():
    sys.path.insert(1, "./dia")
    from dia.model import DEFAULT_SAMPLE_RATE, Dia  # type: ignore

    stdout = sys.stdout
    sys.stdout = sys.stderr
    device = torch.device("cuda")

    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--turns", type=int, default=2)
    parser.add_argument("-c", "--context", type=int, default=2)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    args = parser.parse_args()

    while True:
        line = sys.stdin.readline()
        items = json.loads(line)
        assert len(items) == 1
        item = items[0]
        speaker_audios = item["speaker_audios"]
        text_prompts = []
        audio_prompts = []

        for speaker_audio in speaker_audios:
            audio, text = get_speaker_audio(speaker_audio, DEFAULT_SAMPLE_RATE)
            print(speaker_audio, audio.shape, text)
            audio_prompts.append(audio)
            text_prompts.append(text)

        todos = deque((idx % len(speaker_audios), turn) for idx, turn in enumerate(item["turns"]))
        context = []
        segments = []
        start = 0.0

        all_wavs = []

        try:
            while todos:
                entries = []
                if args.context:
                    context = context[-args.context :]
                else:
                    context = []
                entries += context
                generating = []
                for _ in range(args.turns):
                    speaker, turn = todos.popleft()
                    generating.append((speaker, turn))
                    entries.append((speaker, turn, None))
                    if not todos:
                        break
                text, audio_tokens = build_audio_text(model, entries, audio_prompts, text_prompts, len(context))
                print("Context: ", " | ".join(turn for _, turn, _ in entries))
                print("Generating turn: ", " | ".join(turn for _, turn in generating))
                trials = 5
                for trial in range(trials):
                    try:
                        output = model.generate(
                            text,
                            audio_prompt=[audio_tokens],
                            cfg_scale=args.cfg_scale,
                            verbose=True,
                        )
                    except RuntimeError:
                        if trial == trials - 1:
                            raise
                        print("Generation of chunk failed, retrying...")
                    else:
                        output = torch.from_numpy(output)[None]
                        output = Smoother()(output)
                        duration = output.shape[-1] / DEFAULT_SAMPLE_RATE
                        if args.turns == 1 or len(speaker_audios) == 1:
                            # When generating more than one turn at a time with more than one speaker,
                            # we cannot know where the change of speaker boundary is.
                            speaker, _ = generating[0]
                            segments.append((speaker, (start, start + duration)))
                            start += duration
                        all_wavs.append(output)
                        for speaker, turn in generating:
                            context.append((speaker, turn, output))
                            output = None
                        break
            output = torch.cat(all_wavs, dim=-1)
            print("SAVING", item["output_file"], output.shape)
            output.clamp(-0.99, 0.99)
            sphn.write_wav(item["output_file"], output.numpy()[0], DEFAULT_SAMPLE_RATE)
            if segments:
                with open(Path(item["output_file"]).with_suffix(".segments.json"), "w") as f:
                    json.dump({"segments": segments}, f)
            stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
            stdout.flush()
        except RuntimeError:
            print("ERROR")
            traceback.print_exc()
            stdout.write("external_tts:" + json.dumps({"status": "failed"}) + "\n")
            stdout.flush()


if __name__ == "__main__":
    main()
