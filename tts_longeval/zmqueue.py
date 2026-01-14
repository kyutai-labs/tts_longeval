# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A ZeroMQ based job queue, with the goal of having no external dependency, e.g. redis.
Not the most robust, but in case something get stucks, one can always restart the whole command."""

import logging
import pickle
import socket
import threading
import time
import typing as tp
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field

import zmq
from pydantic import BaseModel

A = tp.TypeVar("A")
logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    try:
        # Gets the default network interface IP
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"  # Fallback to localhost


class _WithExitStack:
    def __init__(self):
        self._stack = ExitStack()

    def __enter__(self) -> tp.Self:
        self._stack.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._stack.__exit__(*exc)


@dataclass
class ServerQueue:
    items: deque = field(default_factory=deque)
    total_count: int = 0


@dataclass
class ServerState:
    queues: dict[str, ServerQueue] = field(default_factory=dict)


class PushRequest(BaseModel):
    queue: str
    item: tp.Any


class PushReply(BaseModel):
    ok: bool = True


class PullRequest(BaseModel):
    queue: str
    num_items: int = 1


class PullReply(BaseModel):
    items: list


class _WithNameAndReqSocket(_WithExitStack):
    """Base class for clients to ZMQueue.
    Args:
        name: name of the queue to interact with.
        address: address of the queue server."""

    def __init__(self, name: str, address: str):
        super().__init__()
        self.name = name
        self.address = address
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 100)
        self.socket.setsockopt(zmq.RCVTIMEO, 30_000)
        self.socket.setsockopt(zmq.SNDTIMEO, 30_000)
        self._initialized = False

    def _ensure(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        logger.debug("Connecting to %s:%s.", self.address, self.name)
        self.socket.connect(self.address)
        self._stack.callback(self._close)

    def _close(self):
        logger.debug("Closing connection to %s:%s.", self.address, self.name)
        self.socket.close()


class Pusher(tp.Generic[A], _WithNameAndReqSocket):
    """Client to `ZMQueue`, can push items to it."""

    def push(self, x: A) -> None:
        self._ensure()
        req = PushRequest(queue=self.name, item=x)
        self.socket.send(pickle.dumps(req))
        raw = self.socket.recv()
        rep = tp.cast(PushReply, pickle.loads(raw))
        assert rep.ok


class Puller(tp.Generic[A], _WithNameAndReqSocket):
    """Client to `ZMQueue`, can pull items from it."""

    def pull(self, num_items: int = 1) -> list[A]:
        self._ensure()
        req = PullRequest(queue=self.name, num_items=num_items)
        self.socket.send(pickle.dumps(req))
        raw = self.socket.recv()
        rep = tp.cast(PullReply, pickle.loads(raw))
        return tp.cast(list[A], rep.items)


class Queue(tp.Generic[A]):
    """Represents a queue. It can get pickled and passed to a worker, which can
    call either `pusher` or `puller` to get a client to it.."""

    def __init__(self, name: str, pull_address: str, push_address: str):
        self.name = name
        self.pull_address = pull_address
        self.push_address = push_address

    def pusher(self) -> Pusher[A]:
        return Pusher(self.name, self.push_address)

    def puller(self) -> Puller[A]:
        return Puller(self.name, self.pull_address)


class ZMQueue(_WithExitStack):
    """ZMQueue service, acting as a server on both a `pull_address` and `push_address`."""

    def __init__(self, pull_address: str, push_address: str = "inproc://zmqueue", log_every_sec: float = 60.0):
        super().__init__()
        self.pull_address = pull_address
        self.push_address = push_address
        self.log_every_sec = log_every_sec

        context = zmq.Context.instance()
        self.for_pull = context.socket(zmq.REP)
        self.for_pull.setsockopt(zmq.LINGER, 100)
        self.for_push = context.socket(zmq.REP)
        self.for_push.setsockopt(zmq.LINGER, 100)
        self.poller = zmq.Poller()
        self._thread = threading.Thread(target=self._run)
        self._closing = False
        self.state = ServerState()

    def __enter__(self):
        super().__enter__()
        logger.info("ZMQueue starting, on pull addr %s, push addr %s.", self.pull_address, self.push_address)
        self.for_pull.bind(self.pull_address)
        self.for_push.bind(self.push_address)
        self.poller.register(self.for_pull, zmq.POLLIN)
        self.poller.register(self.for_push, zmq.POLLIN)
        self._thread.start()
        self._stack.callback(self._close)
        return self

    def _close(self):
        logger.debug("ZMQueue closing.")
        self._closing = True
        self._thread.join()
        self.for_pull.close()
        self.for_push.close()
        logger.info("ZMQueue closed.")

    def _run(self):
        logger.debug("ZMQueue thread started.")
        last_log = time.time()
        while not self._closing:
            events = dict(self.poller.poll(100))
            if self.for_pull in events:
                raw = self.for_pull.recv()
                pullreq = tp.cast(PullRequest, pickle.loads(raw))
                pullrep = self._handle_pull(pullreq)
                self.for_pull.send(pickle.dumps(pullrep))
            if self.for_push in events:
                raw = self.for_push.recv()
                pushreq = tp.cast(PushRequest, pickle.loads(raw))
                pushrep = self._handle_push(pushreq)
                self.for_push.send(pickle.dumps(pushrep))
            if time.time() - last_log > self.log_every_sec:
                self._log()
                last_log = time.time()

    def _log(self):
        cols = []
        for name, queue in self.state.queues.items():
            if queue.items:
                total = queue.total_count
                remaining = len(queue.items)
                cols.append(f"{name:>40}: {remaining: 6d} / {total: 6d}")
        per_row = 3
        rows = []
        for offset in range(0, len(cols), per_row):
            rows.append(", ".join(cols[offset : offset + per_row]))
        usage = "\n".join(rows)
        logger.info("ZMQueue usage:\n%s", usage)

    def _handle_pull(self, request: PullRequest) -> PullReply:
        name = request.queue
        try:
            queue = self.state.queues[name].items
        except KeyError:
            return PullReply(items=[])
        items = []
        for _ in range(request.num_items):
            if not queue:
                break
            items.append(queue.popleft())
        return PullReply(items=items)

    def _handle_push(self, request: PushRequest) -> PushReply:
        if request.queue not in self.state.queues:
            self.state.queues[request.queue] = ServerQueue()
        queue = self.state.queues[request.queue]
        queue.items.append(request.item)
        queue.total_count += 1
        return PushReply()

    def new_queue(self, name: str) -> Queue:
        """Declare a new named queue on the service."""
        prefix = "tcp://*:"
        if self.pull_address.startswith(prefix):
            local_ip = get_local_ip()
            real_pull_address = f"tcp://{local_ip}:" + self.pull_address.removeprefix(prefix)
        else:
            real_pull_address = self.pull_address

        return Queue(name, real_pull_address, self.push_address)
