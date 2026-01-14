# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Speaker similarity metric."""

import logging
from pathlib import Path
import json

from pydantic import BaseModel
import sphn
import torch

from .data import Sample
from .loadable import Loadable
from .task import BatchedTask
from .utils import write_and_rename
from .wavlm import load_wavlm_speaker_model, WAVLM_SPEAKER_SIM_PATH
from external_tools.speaker import hf_get


logger = logging.getLogger(__name__)


def get_speaker_sims(file: Path, quantiles: list[float]) -> dict[str, float] | None:
    """Search for a pre-computed result file for the speaker similarity, and return a dict of metrics.

    Args:
        file: generated audio file path.
        quantiles: list of quantiles over which to compute the metrics, where the quantiles refer
            to chunks over the total duration of the generated audio (e.g. first 25% seconds, etc.).
    Returns:
        dict with keys 'sim', 'nn' (for nearest neighbor, is the segment embedding nearer to
            the expected speaker than the other one). For per quantile, names are shorten
            to 's_q{quantile}', and 'n_q{quantile}'.
    """
    result_file = file.with_suffix(".speaker.json")
    if not result_file.exists():
        logger.debug("Missing speaker similarity result file: %s", result_file)
        return None
    try:
        result = SpeakerResult.model_validate_json(result_file.read_text())
    except Exception:
        logger.error("Exception while parsing file %s", result_file)
        raise
    if not result.results:
        logger.warning("Speaker similarity result file is empty: %s", result_file)
        return None
    total = result.result_for_segment()
    metrics = {
        "sim": total.similarity,
        "nn": total.is_best,
    }
    duration = result.results[-1].end
    start = 0.0
    for quantile in quantiles:
        end = quantile * duration
        try:
            chunk = result.result_for_segment((start, end))
        except ValueError:
            logger.warning("Empty segment for quantile %f for file %s", quantile, result_file)
        else:
            metrics.update(
                {
                    f"s_q{quantile}": chunk.similarity,
                    f"n_q{quantile}": chunk.is_best,
                }
            )
        start = end
    return metrics


class SpeakerSegmentResult(BaseModel):
    """Result for one segment.
    Args:
        start: start timestamp.
        end: end timestamp.
        similarity: cosine similarity between the segment embedding and the expected one.
        is_best: is the expected speaker nearest to the embedding.
        speaker: index of the expected speaker, if known.
    """

    start: float
    end: float
    similarity: float
    is_best: float
    speaker: int | None = None


class SpeakerResult(BaseModel):
    """Results of the speaker similarity metric over multiple segments."""

    results: list[SpeakerSegmentResult]

    def result_for_segment(self, segment: tuple[float, float] | None = None) -> SpeakerSegmentResult:
        if segment is None:
            segment = self.results[0].start, self.results[-1].end

        total_sim = 0.0
        total_best = 0.0
        total = 0.0
        for result in self.results:
            overlap = min(segment[1], result.end) - max(segment[0], result.start)
            if overlap <= 0:
                continue
            total += overlap
            total_sim += result.similarity * overlap
            total_best += result.is_best * overlap
        if total == 0.0:
            raise ValueError("No data on this segment.")
        return SpeakerSegmentResult(
            start=segment[0], end=segment[1], similarity=total_sim / total, is_best=total_best / total
        )


class SpeakerSimilarity:
    """Compute the speaker similarity metrics over segments. A new segment is processed upon
    a change of speaker, or within a single speaker turn, based on `min_segment_duration` and
    `max_segment_duration`.

    Args:
        model_path: path to the speaker embedding model.
        min_segment_duration: will ignore any segment less than this duration.
        max_segment_duration: will split a segment if longer than this duration.
    """

    def __init__(self, model_path: Path, min_segment_duration: float, max_segment_duration: float):
        self.device = torch.device("cuda")
        self.model = load_wavlm_speaker_model(model_path, device=self.device)
        self.max_segment_duration = max_segment_duration
        self.min_segment_duration = min_segment_duration

    @property
    def sample_rate(self) -> int:
        return 16000

    def close(self):
        pass

    def _get_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.dim() == 2
        audio = audio.mean(dim=0, keepdim=True)
        return self.model(audio.to(self.device))

    def get_result(self, speaker_audios: list[Path], audio_file: Path) -> SpeakerResult | None:
        wav_np, sr = sphn.read(audio_file, sample_rate=self.sample_rate)
        wav = torch.from_numpy(wav_np)
        segments: list[tuple[int, tuple[float, float]]]
        if len(speaker_audios) == 1:
            segments = [(0, (0.0, wav.shape[-1] / sr))]
        else:
            segment_file = audio_file.with_suffix(".segments.json")
            if not segment_file.exists():
                return None
            segments = json.loads(segment_file.read_text())["segments"]

        speaker_embs = torch.cat(
            [
                self._get_embedding(torch.from_numpy(sphn.read(speaker_audio, sample_rate=self.sample_rate)[0]))
                for speaker_audio in speaker_audios
            ]
        )

        results = []
        for speaker, (seg_start, seg_end) in segments:
            while seg_start < seg_end:
                this_start = seg_start
                this_end = min(seg_end, this_start + self.max_segment_duration)
                seg_start += self.max_segment_duration
                if this_end - this_start < self.min_segment_duration:
                    continue
                wav_segment = wav[..., int(sr * this_start) : int(sr * this_end)]
                if wav_segment.shape[0] == 2:
                    wav_segment = wav_segment[speaker][None]
                emb = self._get_embedding(wav_segment)

                all_sims = torch.nn.functional.cosine_similarity(emb, speaker_embs)
                sim = all_sims[speaker].item()
                is_best = float(all_sims.argmax(dim=0).item() == speaker)

                results.append(
                    SpeakerSegmentResult(
                        start=this_start, end=this_end, similarity=sim, is_best=is_best, speaker=speaker
                    )
                )
        return SpeakerResult(results=results)


class SpeakerSimilarityConfig(BaseModel, Loadable[SpeakerSimilarity]):
    model_path: Path = WAVLM_SPEAKER_SIM_PATH
    max_segment_duration: float = 30.0
    min_segment_duration: float = 1.0

    def get(self) -> SpeakerSimilarity:
        return SpeakerSimilarity(self.model_path, self.min_segment_duration, self.max_segment_duration)


class SpeakerSimilarityTask(BatchedTask[SpeakerSimilarity, tuple[Sample, Path]]):
    def __init__(self, debug: bool):
        self.debug = debug

    def __call__(self, loaded: SpeakerSimilarity, args: list[tuple[Sample, Path]]):
        for sample, file in args:
            speaker_audios = [hf_get(speaker_audio) for speaker_audio in sample.speaker_audios]
            result = loaded.get_result(speaker_audios, file)
            if result is None:
                continue
            with write_and_rename(file.with_suffix(".speaker.json"), "w") as fout:
                fout.write(result.model_dump_json())
