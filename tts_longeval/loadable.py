# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Simple protocol, a `Loadable` is something that can be loaded in a worker job, and is expected
to take some time to do so, e.g. a model. The loadeded value should be `Closable`, e.g. have a close method.
"""

from abc import ABC, abstractmethod
import typing as tp


L = tp.TypeVar("L", bound="Closable")
R = tp.TypeVar("R")


class Closable(tp.Protocol):
    def close(self) -> None: ...


class Loadable(ABC, tp.Generic[L]):
    @abstractmethod
    def get(self) -> L: ...
