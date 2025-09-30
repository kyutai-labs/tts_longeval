import json
import logging
from pathlib import Path

import numpy as np
import sphn
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors

from tts_longeval.data import Sample
from tts_longeval.loadable import Loadable
from tts_longeval.task import BatchedTask
from tts_longeval.utils import write_and_rename

logger = logging.getLogger(__name__)


def get_drift_metrics(file: Path, quantiles: list[float]) -> dict[str, float] | None:
    """Compute drift metrics from a file containing normalized distances.

    Args:
        file: Path to the .drift.json file.
        quantiles: List of quantiles (floats between 0 and 1) to split the data.

    Returns:
        Dictionary with 'mean' for the whole file, and 'q{quantile}' for each quantile chunk.
    """
    result_file = file.with_suffix(".drift.json")
    if not result_file.exists():
        print()
        return None

    try:
        data = json.loads(result_file.read_text())
        distances = data["normalized_distances"]
    except Exception:
        logger.error("Exception while parsing file %s", result_file)
        raise

    metrics = {"drift": float(np.mean(distances))}

    for start, end in zip([0.0] + quantiles[:-1], quantiles, strict=True):
        start_i = int(start * len(distances))
        end_i = int(end * len(distances))
        metrics[f"d_q{end}"] = float(np.mean(distances[start_i:end_i]))

    return metrics


def get_normalized_distances(latents: torch.Tensor, training_prefix_frames: int):
    latents_np = latents[0].detach().cpu().numpy().T

    nn = NearestNeighbors()
    nn.fit(latents_np[:training_prefix_frames, :])
    nearest_distances = nn.kneighbors(latents_np, n_neighbors=5)[0]
    nearest_distances = nearest_distances.mean(axis=1)
    nearest_distances /= nearest_distances[:training_prefix_frames].mean()

    return nearest_distances


class Drift:
    def __init__(self, train_prefix_duration: float):
        self.train_prefix_duration = train_prefix_duration

        self.device = "cuda"
        checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
        self.mimi = checkpoint_info.get_mimi(device=self.device)

    def get_result(self, file: Path):
        audio, _sr = sphn.read(file, sample_rate=24000)

        with torch.no_grad():
            latents = self.mimi.encode_to_latent(
                torch.Tensor(audio[None, :, :]).to(self.device)
            )

        normalized_distances = get_normalized_distances(
            latents,
            training_prefix_frames=int(
                self.mimi.frame_rate * self.train_prefix_duration
            ),
        )
        return list(normalized_distances)

    def close(self):
        pass


class DriftConfig(BaseModel, Loadable[Drift]):
    train_prefix_duration: float = 30.0

    def get(self) -> Drift:
        return Drift(self.train_prefix_duration)


class DriftTask(BatchedTask[Drift, tuple[Sample, Path]]):
    def __init__(self, debug: bool):
        self.debug = debug

    def __call__(self, loaded: Drift, args: list[tuple[Sample, Path]]):
        # It would be possible to *actually* batch this to be more efficient,
        # but it's fast enough that we don't care
        for _sample, file in args:
            normalized_distances = loaded.get_result(file)
            if normalized_distances is None:
                continue
            with write_and_rename(file.with_suffix(".drift.json"), "w") as fout:
                json.dump({"normalized_distances": normalized_distances}, fout)
