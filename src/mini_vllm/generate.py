"""Greedy generation, two ways: with the KV-cache (fast path) and without it
(reference). They must produce identical tokens — that's the correctness test
for the cache.
"""

from __future__ import annotations

import torch

from .model import KVCache, TinyTransformer


@torch.no_grad()
def generate_no_cache(
    model: TinyTransformer, input_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """Reference decode: re-run the full sequence every step (O(n^2), simple)."""
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = model(ids)                 # full forward over the whole prefix
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate_cached(
    model: TinyTransformer, input_ids: torch.Tensor, max_new_tokens: int
) -> torch.Tensor:
    """Cached decode: prefill once, then feed one token at a time (O(n) / step)."""
    cache = KVCache.make(model.cfg.n_layer)
    # prefill: process the whole prompt, populating the cache
    logits = model(input_ids, cache=cache)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    ids = torch.cat([input_ids, next_id], dim=1)

    for _ in range(max_new_tokens - 1):
        # decode: feed ONLY the last token; attention sees the rest via the cache
        logits = model(next_id, cache=cache)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids
