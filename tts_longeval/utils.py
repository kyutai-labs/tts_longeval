# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities."""

import logging
import os
import sys
import time
import typing as tp
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import httpx

T = tp.TypeVar("T")
logger = logging.getLogger(__name__)


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
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])


def retry(max_retries: int = 3, delay: float = 1.0) -> tp.Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, ConnectionResetError, httpx.ConnectError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Connection error on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"Failed after {max_retries + 1} attempts: {e}")
            raise last_exception  # type: ignore

        return tp.cast(F, wrapper)

    return decorator
