# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A task is something that requires something heavy to load,
a `Loadable`, and applies it so some number of items.
"""
from abc import ABC, abstractmethod
from contextlib import ExitStack
import logging
import random
import os
import typing as tp

from .loadable import Loadable, L
from .zmqueue import Queue, A
from .utils import init_logging


logger = logging.getLogger(__name__)


class BatchedTask(ABC, tp.Generic[L, A]):
    @abstractmethod
    def __call__(self, loaded: L, args: list[A]) -> None:
        ...


class Tasker(tp.Generic[L, A]):
    """Applies a task, ensuring that `loadable` is loaded only once,
    and its resources are closed at the end. The task items are popped from
    `queue`."""
    def __init__(self, max_batch_size: int, task: BatchedTask[L, A], loadable: Loadable[L], queue: Queue[A]):
        self.max_batch_size = max_batch_size
        self.task = task
        self.loadable = loadable
        self.queue = queue

    def run(self) -> None:
        loaded: L | None = None
        with ExitStack() as stack, self.queue.puller() as puller:
            while True:
                logger.debug("Asking for batch at %s", puller.name)
                batch = puller.pull(self.max_batch_size)
                if not batch:
                    break
                if loaded is None:
                    loaded = self.loadable.get()
                    stack.callback(loaded.close)
                self.task(loaded, batch)
                batch.clear()


class MultiTasker:
    def __init__(self, taskers: list[Tasker], should_init_logging: bool = True):
        self.taskers = taskers
        self.should_init_logging = should_init_logging

    def run(self) -> None:
        if self.should_init_logging:
            init_logging(verbose=bool(os.environ.get('_TTS_LONGEVAL_VERBOSE')))
        random.shuffle(self.taskers)
        for tasker in self.taskers:
            tasker.run()

    def __bool__(self) -> bool:
        return bool(self.taskers)
