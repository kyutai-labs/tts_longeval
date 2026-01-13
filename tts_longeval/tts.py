# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Definition of TTS models and tasks. There are two kind of TTS: either API based
(Elevenlabs) or subprocess based, using a protocol to feed it the script to generate.
This allows a good isolation of the dependencies between TTS engines, otherwise, it would
be a complete hell to support all of them in a single codebase."""
from abc import ABC, abstractmethod
from functools import cache
import json
import logging
import os
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import typing as tp

from pydantic import BaseModel
import sphn
import torch

from external_tools.audio import Smoother
from .data import Sample
from .loadable import Loadable
from .task import BatchedTask


TTS = tp.TypeVar('TTS', bound='BaseTTS')
logger = logging.getLogger(__name__)


class LoadableTTS(Loadable[TTS]):
    @property
    @abstractmethod
    def is_api(self) -> bool:
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        ...

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        ...

    @property
    @abstractmethod
    def need_tags(self) -> list[str]:
        ...


class BaseTTS(ABC):
    def generate_batch(self, samples: list[Sample], output_files: list[Path]) -> None:
        ...

    def close(self) -> None:
        pass


class LoadableExternalTTS(LoadableTTS['ExternalTTS']):
    """An external TTS, running in a subprocess following the prococol.

    Args:
        id: name for the TTS.
        command: subprocess command to run.
        cwd: working directory for the subprocess.
        is_api: true if it is API based, e.g. should run in a local CPU thread.
        max_batch_size: maximum batch size supported by this external TTS engine.
        supported_languages: languages that the TTS engine supports.
        need_tags: set of tags that a data sample must have to be processed. This is used for
            instance to indicate TTS that can only support single speaker samples.
    """
    def __init__(self, id: str, command: list[str], cwd: Path, is_api: bool, max_batch_size: int,
                 supported_languages: list[str], need_tags: list[str]):
        self.command = command
        self.cwd = cwd
        self._id = id
        self._is_api = is_api
        self._max_batch_size = max_batch_size
        self._supported_languages = supported_languages
        self._need_tags = need_tags

    @property
    def is_api(self) -> bool:
        return self._is_api

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @property
    def id(self):
        return self._id

    @property
    def supported_languages(self) -> list[str]:
        return self._supported_languages

    @property
    def need_tags(self) -> list[str]:
        return self._need_tags

    def get(self) -> 'ExternalTTS':
        return ExternalTTS(self.command, self.cwd)


class ExternalTTS(BaseTTS):
    """
    This is an instance of a TTS engine running in a subprocess. It should follow the *protocol*:
    - one JSONL encoded single line is fed through STDIN (see the keys in `generate_batch`).
    - the subprocess should output one line starting with `external_tts:` followed by a simple
        JSONL reply indicating the status of the generation. Any other output line without this prefix
        is just printed out.
    """
    def __init__(self, command: list[str], cwd: Path):
        self._proc = sp.Popen(command, stdin=sp.PIPE, stdout=sp.PIPE, cwd=cwd, text=True, bufsize=1)

    def close(self):
        if self._proc.returncode is None:
            self._proc.terminate()
            self._proc.wait()

    def generate_batch(self, samples: list[Sample], output_files: list[Path]) -> None:
        batch = []
        for sample, output_file in zip(samples, output_files):
            item = {
                'turns': sample.turns,
                'speaker_audios': sample.speaker_audios,
                'language': sample.language,
                'output_file': str(output_file.absolute()),
                'extra': sample.model_extra,
            }
            batch.append(item)
        batch_json = json.dumps(batch)
        if self._proc.poll() is not None:
            raise RuntimeError(f"Process has exited with code {self._proc.returncode}")
        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(batch_json)
        self._proc.stdin.write('\n')
        if self._proc.poll() is not None:
            raise RuntimeError(f"Process has exited with code {self._proc.returncode}")
        while True:
            line = self._proc.stdout.readline()
            if self._proc.poll() is not None:
                raise RuntimeError(f"Process has exited with code {self._proc.returncode}")
            if line.startswith('external_tts:'):
                result = json.loads(line.split(':', 1)[1])
                if result['status'] == 'failed':
                    raise RuntimeError("Generation failed.")
                assert result['status'] == 'ok'
                break
            else:
                print(line)


class ExternalTTSConfig(BaseModel):
    """See `LoadableExternalTTS`."""
    command: list[str]
    cwd: Path
    is_api: bool = False
    max_batch_size: int = 1
    active: bool = True
    supported_languages: list[str] = ['en']
    need_tags: list[str] = []

    def get(self, id: str) -> LoadableExternalTTS | None:
        if not self.active:
            return None
        return LoadableExternalTTS(id, self.command, self.cwd, self.is_api, self.max_batch_size,
                                   self.supported_languages, self.need_tags)


class TTSTask(BatchedTask[TTS, tuple[Sample, Path]]):
    def __init__(self, debug: bool):
        self.debug = debug

    def __call__(self, loaded: TTS, args: list[tuple[Sample, Path]]):
        samples = [samples for samples, _ in args]
        output_files = [output_file for _, output_file in args]
        begin = time.time()
        try:
            loaded.generate_batch(samples, output_files)
        except Exception as exc:
            sample_ids = ', '.join(x.id for x in samples)
            logger.error(f"Error generating batch with model {loaded.__class__}, samples {sample_ids}: {exc}")
            if self.debug:
                raise
        else:
            end = time.time()
            duration = (end - begin) / len(samples)
            for output_file in output_files:
                info = {
                    'duration': duration
                }
                output_file.with_suffix('.done').write_text(json.dumps(info))


class LoadableElevenAPI(LoadableTTS['LoadableElevenAPI']):
    """An API based TTS, using ElevenLabs. Note that we will look for a voice with the
    last part of the speaker audio filename in the available voices."""
    def __init__(self, id: str, model_id: str, supported_languages: list[str], need_tags: list[str]):
        from elevenlabs.client import ElevenLabs
        api_key = os.environ['ELEVENLAB_API_KEY']
        self.client = ElevenLabs(api_key=api_key)

        self.model_id = model_id
        self._id = id
        self._supported_languages = supported_languages
        self._need_tags = need_tags

    def close(self):
        pass

    def get(self) -> 'LoadableElevenAPI':
        return self

    @property
    def max_batch_size(self):
        return 1

    @property
    def is_api(self) -> bool:
        return True

    @property
    def id(self):
        return self._id

    @property
    def supported_languages(self) -> list[str]:
        return self._supported_languages

    @property
    def need_tags(self) -> list[str]:
        return self._need_tags

    @property
    @cache
    def voice_ids(self) -> dict[str, str]:
        out = {}
        for voice in self.client.voices.get_all().voices:
            out[voice.name] = voice.voice_id
        return out

    def _gen_one(self, voice_name: str, text: str) -> tuple[torch.Tensor, int]:
        try:
            voice_id = self.voice_ids[voice_name]
        except KeyError:
            raise ValueError(f"Invalid voice {voice_name}")

        logger.info("11Lab generating with %s, voice %s: %s", self.model_id, voice_id, text)
        result = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )

        with NamedTemporaryFile(suffix=".mp3") as file:
            for chunk in result:
                file.write(chunk)
            file.flush()
            wav, sr = sphn.read(file.name)
            wav = torch.from_numpy(wav)
            wav = Smoother()(wav)
        return wav, sr

    def generate_batch(self, samples: list[Sample], output_files: list[Path]) -> None:
        assert len(samples) == 1
        sample = samples[0]
        all_segments = []
        segments = []
        out_sample_rate = None
        if len(sample.speaker_audios) == 1:
            voice_name = sample.speaker_audios[0].rsplit('/', 1)[-1]
            text = " ".join(sample.turns)
            wav, out_sample_rate = self._gen_one(voice_name, text)
            all_segments.append(wav)
        else:
            start = 0
            for idx, turn in enumerate(sample.turns):
                speaker_idx = idx % len(sample.speaker_audios)
                voice_name = sample.speaker_audios[speaker_idx].rsplit('/', 1)[-1]
                wav, sr = self._gen_one(voice_name, turn)
                if out_sample_rate is None:
                    out_sample_rate = sr
                else:
                    assert out_sample_rate == sr
                duration = wav.shape[-1] / sr
                segments.append((speaker_idx, (start, start + duration)))
                start += duration
                all_segments.append(wav)
        assert out_sample_rate is not None
        wav = torch.cat(all_segments, dim=-1)
        wav.clamp_(-0.99, 0.999)
        sphn.write_wav(output_files[0], wav.numpy(), out_sample_rate)
        if segments:
            with open(output_files[0].with_suffix('.segments.json'), 'w') as f:
                json.dump({'segments': segments}, f)
        output_files[0].with_suffix('.done').touch()


class ElevenAPIConfig(BaseModel):
    model_id: str
    active: bool = True
    supported_languages: list[str] = ['en', 'fr']
    need_tags: list[str] = []

    def get(self, id: str) -> LoadableElevenAPI | None:
        if not self.active:
            return None
        return LoadableElevenAPI(id, self.model_id, self.supported_languages, self.need_tags)
