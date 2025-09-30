# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main config definition for the TOML files."""

from pathlib import Path

from pydantic import BaseModel, Field

from tts_longeval.drift import DriftConfig

from .asr import ASRConfig
from .data import DatasetConfig
from .metrics import MetricsConfig
from .runner import RunnerConfig
from .speakersim import SpeakerSimilarityConfig
from .tts import ElevenAPIConfig, ExternalTTSConfig


class MainConfig(BaseModel):
    output_folder: Path
    debug: bool = False
    queue_addr: str = "tcp://*:34873"


class Config(BaseModel):
    main: MainConfig
    asr: ASRConfig
    dataset: DatasetConfig
    runner: RunnerConfig
    tts: dict[str, ExternalTTSConfig] = {}
    tts11: dict[str, ElevenAPIConfig] = {}
    speakersim: SpeakerSimilarityConfig = Field(default_factory=SpeakerSimilarityConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
