# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main config definition for the TOML files."""

from pathlib import Path

from pydantic import BaseModel, Field

from .asr import ASRConfig
from .data import DatasetConfig
from .metrics import MetricsConfig
from .runner import RunnerConfig
from .speakersim import SpeakerSimilarityConfig
from .tts import CartesiaAPIConfig, ElevenAPIConfig, ExternalTTSConfig, GradiumAPIConfig


class MainConfig(BaseModel):
    output_folder: Path
    debug: bool = False
    queue_addr: str = "tcp://*:34873"
    output_format: str = "wav"


class Config(BaseModel):
    main: MainConfig
    asr: ASRConfig
    dataset: DatasetConfig
    runner: RunnerConfig
    tts: dict[str, ExternalTTSConfig] = {}
    tts11: dict[str, ElevenAPIConfig] = {}
    ttscartesia: dict[str, CartesiaAPIConfig] = {}
    ttsgradium: dict[str, GradiumAPIConfig] = {}
    speakersim: SpeakerSimilarityConfig = Field(default_factory=SpeakerSimilarityConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
