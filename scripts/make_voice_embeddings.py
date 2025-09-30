import argparse
import re
import subprocess
import tomllib
from pathlib import Path

from tts_longeval.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    raw = tomllib.load(args.config.open("rb"))
    config = Config.model_validate(raw)

    for tts_name, tts_config in config.tts.items():
        command = tts_config.command
        if command[:3] != ["uv", "run", "external_tts_dsm.py"]:
            print("Skipping non-DSM TTS:", tts_name, command)
            continue

        model_root = None
        for entry in command:
            match = re.match(r"--config=(.*)config\.json$", entry)
            if match:
                model_root = match.group(1)
                break

        if model_root is None:
            print(f"Skipping, could not find --config in command: {command}")
            continue

        print(model_root, "\n")

        command = [
            "uv",
            "run",
            "--with=/home/vaclav/prog/moshi/moshi,julius,torchaudio",
            "/home/vaclav/prog/moshi/scripts/tts_make_voice.py",
            "--model-root",
            model_root,
            "/lustre/scwpod02/client/kyutai/vaclav/data/tts_longform_debug/voices_tts_longeval/ntrex_dialogs",
        ]
        print("Running command:")
        print(" ".join(command), "\n")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
