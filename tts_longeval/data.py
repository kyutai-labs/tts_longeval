# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definition of the Sample schema, used in the dataset JSONL files."""
from pathlib import Path
import re
from glob import glob

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .utils import get_root


def clean_text(text: str):
    """We share some replacement of strange characters at the source of the dataset."""
    text = text.strip()
    text = text.replace('’', "'")
    text = text.replace(':', " ")
    text = text.replace('(', " ")
    text = text.replace(')', " ")
    text = text.replace('\xad', "")
    text = re.sub(r'\s+', ' ', text)
    return text


class Sample(BaseModel):
    """
    Represent a script for a monologue or dialog (or more speakers in the future).

    Args:
        id: used as the filename (and folder if contains a '/'). Make sure to have no weird characters here.
        turns: One turn is either a turn of speech for a dialog, starting with the first speaker
            in `speaker_audios`, then alternating. For monologues, the script can contain "turns", which
            are just arbitrary split of the full content, useful for TTS than cannot process long inputs.
        speaker_audios: path to the speaker audio conditioning files. Please use '.wav' files for maximum
            compatibility. For some TTS engine, a '.txt' with the transcript of the audio should exist.
            In general, prefer relative path and set the root in the TOML files.
        language: expected language for the script, used mostly for the ASR to compute a WER.
        tags: arbitrary set of tags. One can then filter based on it.
    """
    model_config = ConfigDict(extra="allow")

    id: str
    turns: list[str]
    speaker_audios: list[str]
    language: str = 'en'
    reference_audio: Path | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator('turns', mode='after')
    @classmethod
    def clean_up_turns(cls, turns: list[str]) -> list[str]:
        return [clean_text(turn) for turn in turns]


class DatasetConfig(BaseModel):
    """Config for the datasets in the TOML config files.

    Args:
        datasets: list of dataset to process in this run. Can contain wildcard '*'. Name of datasets
            should not contain the '.jsonl' extension.
        speaker_audio_root: root path in which the voices are stored. Can also be a HF dataset with
            the syntax `'hf-dataset://username/dataset_name'`.
    """
    datasets: list[str]
    speaker_audio_root: str = 'hf-dataset://kyutai/voices_tts_longeval'

    @field_validator('datasets', mode='after')
    @classmethod
    def collect_datasets(cls, datasets: list[str]) -> list[str]:
        base_dir = get_root() / 'datasets'
        collected_datasets = []

        for dataset in datasets:
            if '*' in dataset:
                matching_files = glob(str(base_dir / dataset))

                # Extract dataset names from file paths
                for file_path in matching_files:
                    dataset_path = Path(file_path).with_suffix(suffix='')  # Remove .jsonl extension
                    dataset_path = dataset_path.relative_to(base_dir)
                    collected_datasets.append(str(dataset_path))
            else:
                # No pattern, add as-is
                collected_datasets.append(dataset)
        return [str(x) for x in collected_datasets]

    def get(self, dataset: str) -> list[Sample]:
        samples = []
        base_dir = get_root() / 'datasets'
        path = base_dir / dataset
        with path.with_suffix('.jsonl').open('r') as file:
            for line in file:
                sample = Sample.model_validate_json(line)
                sample.speaker_audios = [join_speaker_audio_root(self.speaker_audio_root, speaker_audio)
                                         for speaker_audio in sample.speaker_audios]
                samples.append(sample)
        return samples


def join_speaker_audio_root(root: str, path: str) -> str:
    if path.startswith('hf://') or path.startswith('hf-dataset://'):
        return path
    elif Path(path).is_absolute():
        return path
    elif root.startswith('hf://') or root.startswith('hf-dataset://'):
        return root.rstrip('/') + '/' + str(path)
    else:
        # Make sure the path is absolute as we might change working directory,
        # when calling subprocess based TTS engines.
        return str((Path(root) / path).resolve())
