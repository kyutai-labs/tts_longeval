# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities."""
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import typing as tp
import sys


T = tp.TypeVar('T')


def get_root() -> Path:
    return Path(__file__).parent.parent


@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp"):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def init_logging(verbose: bool = False):
    logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG if verbose else logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
        datefmt="%m-%d %H:%M:%S",
        force=True)
