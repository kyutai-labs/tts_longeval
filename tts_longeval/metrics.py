# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration, aggregation, and display of the metrics."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
import typing as tp

from pydantic import BaseModel, Field
import treetable as tt

from .data import Sample
from .wer import get_wers, get_normalizer
from .speakersim import get_speaker_sims


logger = logging.getLogger(__name__)


class MetricsConfig(BaseModel):
    """Configuration for the calculation of metrics.
    Args:
        workers: number of local worker process to use.
        quantiles: quantiles over which to compute partial metrics, either
            with respect to duration (e.g. extract with the first 25% of seconds) or
            text length (for the WER).
        fallbacks: allow to define for one method, a number of other methods that will be used
            if the primary method failed. Not really tested.
        normalizer: text normalizer for the WER.

        """
    workers: int = 1
    quantiles: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75, 1.])
    fallbacks: dict[str, list[str]] = {}
    normalizer: str = 'whisper'

    def get_pool(self) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(self.workers)


def collect_metrics(config: MetricsConfig, sample: Sample, files: list[Path]) -> dict[str, dict[str, float]]:
    normalizer = get_normalizer(config.normalizer, sample.language)
    for file in files:
        if not file.exists():
            continue
        metrics = {}
        try:
            wers = get_wers(sample, file, config.quantiles, normalizer)
            if wers is None:
                wers = {}
            metrics['wers'] = wers

            speaker_sims = get_speaker_sims(file, config.quantiles)
            if speaker_sims is None:
                speaker_sims = {}
            metrics['spks'] = speaker_sims
        except Exception:
            logger.error("Error while loading metrics for file %s", file)
            raise
        return metrics
    logger.warning("Failed to find metrics for files: %r", files)
    return {}


def average(all_metrics: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    transposed: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for metrics in all_metrics:
        for section, sub_metrics in metrics.items():
            for name, value in sub_metrics.items():
                transposed[section][name].append(value)
    out = {}
    for section, sub_metrics in transposed.items():
        out[section] = {}
        for name, values in sub_metrics.items():
            out[section][name] = sum(values) / len(values)
    return out


def collect_results_for_dataset(
        pool: ProcessPoolExecutor,
        config: MetricsConfig,
        pairs_per_method: dict[str, list[tuple[Sample, Path]]]) -> dict[str, dict[str, dict[str, float]]]:

    samples = {}
    ignored_methods = set()
    for fallbacks in config.fallbacks.values():
        for fallback in fallbacks:
            ignored_methods.add(fallback)

    per_sample_then_method: dict[str, dict[str, Path]] = defaultdict(dict)

    for method, pairs in pairs_per_method.items():
        for sample, file in pairs:
            samples[sample.id] = sample
            per_sample_then_method[sample.id][method] = file

    pendings = []
    for sample_id, per_methods in per_sample_then_method.items():
        sample = samples[sample_id]
        for method, file in per_methods.items():
            if method in ignored_methods:
                continue
            files = [file]
            if method in config.fallbacks:
                for fallback in config.fallbacks:
                    files.append(per_methods[fallback])
            pending = pool.submit(collect_metrics, config, sample, files)
            pendings.append((method, pending))

    metrics_per_method = defaultdict(list)
    for method, pending in pendings:
        metrics_per_method[method].append(pending.result())

    out = {}
    for method, metrics in metrics_per_method.items():
        out[method] = average(metrics)
    return out


def print_results_for_dataset(
        pool: ProcessPoolExecutor,
        config: MetricsConfig,
        pairs_per_method: dict[str, list[tuple[Sample, Path]]]) -> list[dict]:
    results = collect_results_for_dataset(pool, config, pairs_per_method)
    lines: list[dict[str, tp.Any]] = []

    metrics_per_section = defaultdict(dict)
    for name, result in results.items():
        line = {}
        line['method'] = name
        line.update(result)
        for section, metrics in result.items():
            metrics_per_section[section].update(metrics)
        lines.append(line)
    lines.sort(key=lambda x: x['method'])

    groups = [
        tt.leaf("method", align="<"),
    ]
    for section, metrics in metrics_per_section.items():
        leafs = []
        for name in sorted(metrics):
            percent_prefixes = [
                "wer", "w_", "sim", "s_", "nn", "n_"
            ]
            fmt = ".3f"
            for prefix in percent_prefixes:
                if name.startswith(prefix):
                    fmt = ".2%"
            leafs.append(tt.leaf(name, format=fmt, align=">"))
        group = tt.group(section, leafs)
        groups.append(group)
    table = tt.table(groups)
    print(tt.treetable(lines, table, colors=["0", "38;5;245"]))
    return lines
