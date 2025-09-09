# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

from huggingface_hub import hf_hub_download
import sphn
import torch

from .audio import Smoother


def hf_get(filename: str | Path) -> Path:
    print("WTF", filename)
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename.removeprefix("hf://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif filename.startswith("hf-dataset://"):
        parts = filename.removeprefix("hf-dataset://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename, repo_type='dataset'))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    else:
        return Path(filename)


def get_speaker_audio(path_or_url: str, sample_rate: int) -> tuple[torch.Tensor, str]:
    wav_path = hf_get(path_or_url)
    wav, _ = sphn.read(wav_path, sample_rate=sample_rate)
    wav = torch.from_numpy(wav).float()
    text_url = path_or_url.rsplit('.', 1)[0] + '.txt'
    text_path = hf_get(text_url)
    text = text_path.read_text(encoding='utf8').strip()
    wav = Smoother()(wav)
    return wav, text
