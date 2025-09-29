import itertools
import json
import random
from pathlib import Path

import regex
import tqdm

TARGET_TEXT_LENGTH = 7500
N_TEXTS = 32

REPO_ROOT = Path(__file__).parents[1]

DATASET_PATHS = {
    "en": Path(
        "/lustre/scwpod02/client/kyutai/datasets/wikibooks/proc/enwikibooks-20231201.jsonl"
    ),
    "fr": Path(
        "/lustre/scwpod02/client/kyutai/datasets/wikibooks/proc/frwikibooks-20231201.jsonl"
    ),
}

SPEAKERS = {
    "en": [
        ("ntrex_dialogs/en/p231_021.wav", "f"),
        ("ntrex_dialogs/en/p244_016.wav", "f"),
        ("ntrex_dialogs/en/p303_214.wav", "f"),
        ("ntrex_dialogs/en/p239_008.wav", "f"),
        ("ntrex_dialogs/en/p347_022.wav", "m"),
        ("ntrex_dialogs/en/p281_005.wav", "m"),
        ("ntrex_dialogs/en/p247_019.wav", "m"),
        ("ntrex_dialogs/en/p245_022.wav", "m"),
    ],
    "fr": [
        ("ntrex_dialogs/fr/10087_11650_000028-0002.wav", "f"),
        ("ntrex_dialogs/fr/10179_11051_000005-0001.wav", "f"),
        ("ntrex_dialogs/fr/1591_1028_000108-0004.wav", "f"),
        ("ntrex_dialogs/fr/12977_10625_000037-0001.wav", "f"),
        ("ntrex_dialogs/fr/1406_1028_000009-0003.wav", "m"),
        ("ntrex_dialogs/fr/4193_3103_000004-0001.wav", "m"),
        ("ntrex_dialogs/fr/2223_1745_000009-0002.wav", "m"),
        ("ntrex_dialogs/fr/296_1028_000022-0001.wav", "m"),
    ],
}


def get_filtered_texts(dataset_path: Path):
    with dataset_path.open("r") as f:
        texts_filtered = []

        for line in (pbar := tqdm.tqdm(f, desc=f"Processing {dataset_path.name}")):
            text = json.loads(line)["text"]

            if len(text) < TARGET_TEXT_LENGTH:
                continue

            text = text[:TARGET_TEXT_LENGTH]

            newline_chars = text.count("\n") + text.count("\r")
            percent_newlines = (newline_chars / len(text) * 100) if len(text) > 0 else 0
            special_chars = len(regex.findall(r"[#\$<>@/\-%\+\*\(\)]", text))
            percent_special = (special_chars / len(text) * 100) if len(text) > 0 else 0

            if percent_special + percent_newlines > 1:
                continue

            # print(text)
            # print(f"Newline chars: {percent_newlines:.2f}%")
            # print(f"Special chars: {percent_special:.2f}%")
            # print("\033[33m" + "-" * 80 + "\033[0m")

            texts_filtered.append(text)
            pbar.set_postfix({"filtered": len(texts_filtered)})

    rng = random.Random(37)  # Shuffle with a fixed seed for reproducibility
    rng.shuffle(texts_filtered)

    return texts_filtered


def text_to_turns(text: str) -> list[str]:
    # Note this discards the last bit if it doesn't end with punctuation
    turns = [
        # Regex explanation: anything (non-greedy),
        # then '.', '!', or '?',
        # followed by a space or newline so that e.g. "1.23" doesn't split the sentence
        p.strip()
        for p in regex.findall(r"(.*?[\.\!\?](?:\s|\n))", text)
        if p.strip()
    ]
    return turns


def main():
    for language in ["fr", "en"]:
        filtered_texts = get_filtered_texts(DATASET_PATHS[language])
        filtered_texts = filtered_texts[:N_TEXTS]
        speakers = SPEAKERS[language]

        dataset = []
        for (i_speaker, (speaker, gender)), (i_text, text) in itertools.product(
            enumerate(speakers), enumerate(filtered_texts)
        ):
            dataset.append(
                {
                    "id": f"{language}_speaker_{i_speaker}_text_{i_text}",
                    "speaker_audios": [speaker],
                    "tags": ["wikibooks", f"gender={gender}", "monologue"],
                    "turns": text_to_turns(text),
                    "language": language,
                }
            )

        with (REPO_ROOT / "datasets" / f"wikibooks_{language}.jsonl").open("w") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(
                f"Wrote {len(dataset)} samples to {REPO_ROOT / 'datasets' / f'wikibooks_{language}.jsonl'}"
            )


if __name__ == "__main__":
    main()
