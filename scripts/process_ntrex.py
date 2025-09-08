# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from pathlib import Path
import json
import random


def main():
    # Clone the repo https://github.com/MicrosoftTranslator/NTREX somewhere.
    parser = argparse.ArgumentParser(description="Process NTREX data.")
    parser.add_argument("-l", "--language", default="eng", help="Language code (default: 'eng').")
    parser.add_argument("-n", "--num-samples", type=int, help="Number of samples to keep.")
    parser.add_argument("--speaker-audio-root", default=Path("dataset_pre/voices"),
                        type=Path, help="Path to the root of speaker audio files.")
    parser.add_argument("ntrex_root", type=Path, help="Path to the root of NTREX.")
    args = parser.parse_args()
    rng = random.Random(1234)

    doc_ids_path = args.ntrex_root / "DOCUMENT_IDS.tsv"
    doc_ids = [x.strip() for x in doc_ids_path.open('r')]

    suffix = ""

    voices = list((args.speaker_audio_root / (args.language[:2] + suffix)).glob("*.wav"))
    voices.sort()

    if args.language == "eng":
        input_file = args.ntrex_root / "NTREX-128" / 'newstest2019-src.eng.txt'
    else:
        input_file = args.ntrex_root / "NTREX-128" / f'newstest2019-ref.{args.language}.txt'

    def _flush():
        nonlocal turns
        if not turns:
            return
        voice = rng.choice(voices).relative_to(args.speaker_audio_root)
        tags = ['ntrex', f'lang={args.language[:2]}', 'monologue']
        # remove the first sentence, the title of the article, as it often repeats the second one.
        turns = turns[1:]
        item = {
            'id': last_id,
            'speaker_audios': [str(voice)],
            'tags': tags,
            'turns': turns,
            'language': args.language[:2],
        }
        lines.append(json.dumps(item) + '\n')
        turns = []

    turns: list[str] = []
    lines: list[str] = []
    last_id: str | None = None
    with input_file.open('r') as infile:
        for idx, line in enumerate(infile):
            line = line.strip()
            doc_id = doc_ids[idx]
            if last_id is None:
                last_id = doc_id
            elif doc_id != last_id:
                _flush()
            last_id = doc_id
            turns.append(line)
        _flush()

    if args.num_samples:
        suffix = suffix + f"_n{args.num_samples}"
        rng = random.Random(1234)
        rng.shuffle(lines)
        lines = lines[:args.num_samples]
    output_path = Path(f'datasets/ntrex{suffix}_{args.language}.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as outfile:
        for line in lines:
            outfile.write(line)


if __name__ == "__main__":
    main()
