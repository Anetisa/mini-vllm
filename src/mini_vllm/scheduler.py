"""Continuous-batching scheduler + engine — the heart of an LLM server.

Static batching waits for every sequence in a batch to finish before starting
new work; since sequences finish at different times, the batch drains and the
GPU idles. **Continuous batching** (iteration-level scheduling) instead admits
and retires requests *between decode steps*: each iteration runs whatever is
currently active, a finished request is removed immediately, and a waiting one
takes its slot — keeping the running set full.

The correctness property (testable on CPU): interleaving requests that started
at different times must not change any individual result — each request's output
equals generating it alone.

Scope note: this implements the *scheduling* (the vLLM idea) with a readable
per-request execution loop. Fusing each iteration's running set into a single
batched PagedAttention kernel is the throughput optimization; the paged cache in
`paged.py` is the memory backend such a version plugs in. The scheduling logic
is identical either way.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import torch

from .model import KVCache, TinyTransformer


class State(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: int
    prompt_ids: list[int]
    max_new_tokens: int
    eos_id: int | None = None
    output_ids: list[int] = field(default_factory=list)
    state: State = State.WAITING
    cache: KVCache | None = None
    prefilled: bool = False

    def is_finished(self) -> bool:
        if len(self.output_ids) >= self.max_new_tokens:
            return True
        if self.eos_id is not None and self.output_ids and self.output_ids[-1] == self.eos_id:
            return True
        return False


class Scheduler:
    """Bounded running set + FIFO waiting queue. Admits while there's capacity."""

    def __init__(self, max_running: int):
        self.max_running = max_running
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        req.state = State.WAITING
        self.waiting.append(req)

    def admit(self) -> None:
        while self.waiting and len(self.running) < self.max_running:
            req = self.waiting.popleft()
            req.state = State.RUNNING
            self.running.append(req)

    def retire_finished(self) -> list[Request]:
        done = [r for r in self.running if r.is_finished()]
        for r in done:
            r.state = State.FINISHED
        self.running = [r for r in self.running if not r.is_finished()]
        return done

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)


class Engine:
    """Ties a model to the scheduler and runs greedy continuous-batched decoding."""

    def __init__(self, model: TinyTransformer, max_running: int = 4, eos_id: int | None = None):
        self.model = model
        self.eos_id = eos_id
        self.scheduler = Scheduler(max_running)
        self._next_id = 0
        self.finished: dict[int, list[int]] = {}
        # simple observability: max concurrent running set seen across the run
        self.max_running_seen = 0

    def add_request(self, prompt_ids: list[int], max_new_tokens: int) -> int:
        req = Request(self._next_id, list(prompt_ids), max_new_tokens, eos_id=self.eos_id)
        self.scheduler.add(req)
        self._next_id += 1
        return req.req_id

    @torch.no_grad()
    def _advance(self, req: Request) -> None:
        """Produce one new token (prefill on first touch, else single-token decode)."""
        if not req.prefilled:
            req.cache = KVCache.make(self.model.cfg.n_layer)
            ids = torch.tensor([req.prompt_ids], dtype=torch.long)
            logits = self.model(ids, cache=req.cache)
            req.prefilled = True
        else:
            last = torch.tensor([[req.output_ids[-1]]], dtype=torch.long)
            logits = self.model(last, cache=req.cache)
        next_id = int(logits[0, -1, :].argmax().item())
        req.output_ids.append(next_id)

    def step(self) -> None:
        """One scheduler iteration: admit, advance the running set, retire finished."""
        self.scheduler.admit()
        self.max_running_seen = max(self.max_running_seen, len(self.scheduler.running))
        for req in self.scheduler.running:
            self._advance(req)
        for req in self.scheduler.retire_finished():
            self.finished[req.req_id] = req.output_ids

    def run(self) -> dict[int, list[int]]:
        """Drive to completion; returns {req_id: generated token ids}."""
        while self.scheduler.has_work:
            self.step()
        return self.finished
