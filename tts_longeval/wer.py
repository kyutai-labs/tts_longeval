# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""WER metric."""

from collections import defaultdict
from functools import cache, partial
import logging
from pathlib import Path
import string
import typing as tp

import jiwer

from .data import Sample
from .normalizers import BasicTextNormalizer, EnglishTextNormalizer, FrenchNormalizer


logger = logging.getLogger(__name__)


def wer_per_quantile(ref: str, hyp: str, quantiles: list[float]) -> list[float]:
    """Compute a WER for each quantile over the reference text length."""
    p = jiwer.process_words(ref, hyp)
    errors = defaultdict(float)
    for ali in p.alignments[0]:
        if ali.type == "equal":
            continue
        elif ali.type in ["substitute", "delete"]:
            for idx in range(ali.ref_start_idx, ali.ref_end_idx):
                errors[idx] += 1.0
        elif ali.type == "insert":
            errors[ali.ref_start_idx] += ali.hyp_end_idx - ali.hyp_start_idx
        else:
            raise ValueError(f"Bad type {ali.type}")
    le_ref = len(p.references[0])
    start = 0
    outs = []
    for q in quantiles:
        end = int(q * le_ref)
        error = sum(v for k, v in errors.items() if k >= start and k < end)
        error /= max(1, (end - start))
        start = end
        outs.append(error)
    return outs


def f5_norm(x: str, seed_mode: bool = False) -> str:
    """Same normalization as F5TTS. Set `seed_mode` for using instead
    the SEED TTS Eval mode (english only)."""
    for a in string.punctuation:
        if seed_mode and a == "'":
            continue
        x = x.replace(a, "")
    while "  " in x:
        x = x.replace("  ", " ")
    x = x.lower()
    return x


@cache
def get_normalizer(name: str, language: str) -> tp.Callable[[str], str]:
    if name == "whisper":
        if language == "fr":
            return FrenchNormalizer()
        elif language == "en":
            return EnglishTextNormalizer()
        else:
            return BasicTextNormalizer(remove_diacritics=True)
    elif name == "f5":
        return f5_norm
    elif name == "seed_en":
        return partial(f5_norm, seed_mode=True)
    else:
        raise ValueError(f"Unknown normalizer {name}.")


def get_wers(
    sample: Sample, audio_file: Path, quantiles: list[float], normalizer: tp.Callable[[str], str]
) -> dict[str, float] | None:
    txt_file = audio_file.with_suffix(".txt")
    if not txt_file.exists():
        logger.debug("Missing transcript file: %s", txt_file)
        return None

    estimate = txt_file.read_text().strip()
    estimate = normalizer(estimate)
    reference = normalizer(" ".join(x.strip() for x in sample.turns))
    wer = jiwer.wer(reference, estimate)
    wers = wer_per_quantile(reference, estimate, quantiles)
    names = ["wer"]
    for q in quantiles:
        names += [f"w_q{q}"]
    return {name: value for name, value in zip(names, [wer] + wers)}
