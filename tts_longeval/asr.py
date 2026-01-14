# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definition of the ASR models and tasks used to compute the WER."""

import json
import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import sphn
import torch
from pydantic import BaseModel
from transformers import pipeline  # type: ignore

from .data import Sample
from .loadable import Loadable
from .task import BatchedTask
from .utils import write_and_rename

logger = logging.Logger(__name__)
ASR = tp.TypeVar("ASR", bound="BaseASR")
AlignedTranscript = list[tuple[str, tuple[float, float]]]


class LoadableASR(Loadable[ASR]):
    pass


class BaseASR(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @abstractmethod
    def transcribe(self, audio: torch.Tensor, language: str) -> AlignedTranscript: ...

    def close(self) -> None:
        pass


class WhisperHFASRConfig(BaseModel, LoadableASR):
    model_name: str

    def get(self) -> "WhisperHFASR":
        return WhisperHFASR(self.model_name)


class WhisperHFASR(BaseASR):
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=f"{model_name}",
            model_kwargs={
                "use_safetensors": True,
                "attn_implementation": "eager",
            },
            torch_dtype=torch.float16,
            return_timestamps="word",
            device=torch.device("cuda"),
        )
        # the following suppresses a warning when setting the language to transcribe.
        assert self.pipeline.generation_config is not None
        self.pipeline.generation_config.forced_decoder_ids = None

    @classmethod
    def from_config(cls, config: WhisperHFASRConfig, **kwargs):
        return cls(config.model_name, **kwargs)

    @property
    def sample_rate(self):
        return 16000

    def transcribe(self, audio: torch.Tensor, language: str = "en") -> AlignedTranscript:
        assert audio.dim() == 2, "Audio must be [C, T]"
        res = self.pipeline(audio.mean(0).cpu().numpy(), generate_kwargs={"task": "transcribe", "language": language})
        assert res is not None
        assert isinstance(res, dict)
        out: list[tuple[str, tuple[float, float]]] = []
        for chunk in res["chunks"]:
            out.append((chunk["text"], chunk["timestamp"]))

        return out


class ASRConfig(BaseModel):
    provider: str = "whisper_hf"
    whisper_hf: WhisperHFASRConfig

    def get(self) -> LoadableASR:
        if self.provider == "whisper_hf":
            return self.whisper_hf
        else:
            raise ValueError(f"Invalid ASR provider {self.provider}")


class ASRTask(BatchedTask[ASR, tuple[Sample, Path]]):
    def __init__(self, debug: bool):
        self.debug = debug

    def __call__(self, loaded: ASR, args: list[tuple[Sample, Path]]):
        for sample, file in args:
            wav, _ = sphn.read(file, sample_rate=loaded.sample_rate)
            wav = torch.from_numpy(wav)
            try:
                chunks = loaded.transcribe(wav, language=sample.language)
            except RuntimeError as exc:
                if self.debug:
                    raise
                logger.error("Error while transcribing %s: %r", file, exc)
                continue
            text = "".join(w for w, _ in chunks).strip()
            file.with_suffix(".txt").write_text(text)
            asr_file = file.with_suffix(".asr.json")
            with write_and_rename(asr_file, "w") as fout:
                json.dump({"chunks": chunks}, fout)
