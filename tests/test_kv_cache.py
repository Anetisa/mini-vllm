"""KV-cache correctness — all on CPU with a tiny random model.

The defining property: the cache is a *speed* optimization, so cached decoding
must produce exactly the same tokens as recomputing the full prefix each step.
"""

import torch

from mini_vllm.generate import generate_cached, generate_no_cache
from mini_vllm.model import KVCache, ModelConfig, TinyTransformer


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, n_layer=3, n_head=4, d_model=64, max_seq_len=128)
    return TinyTransformer(cfg).eval()


def test_cached_equals_uncached_greedy():
    """The whole point: KV-cache must not change the generated tokens."""
    model = _model()
    prompt = torch.randint(0, 64, (1, 5))
    a = generate_no_cache(model, prompt, max_new_tokens=20)
    b = generate_cached(model, prompt, max_new_tokens=20)
    assert torch.equal(a, b), "cached decode diverged from the reference"


def test_cached_equals_uncached_batch():
    """Works for a batch of prompts of equal length."""
    model = _model(1)
    prompt = torch.randint(0, 64, (4, 6))
    a = generate_no_cache(model, prompt, max_new_tokens=15)
    b = generate_cached(model, prompt, max_new_tokens=15)
    assert torch.equal(a, b)


def test_prefill_then_decode_matches_full_forward():
    """One-shot forward over a sequence == prefill + incremental steps (logits)."""
    model = _model(2)
    ids = torch.randint(0, 64, (1, 8))

    full = model(ids)                       # [1, 8, V] no cache

    cache = KVCache.make(model.cfg.n_layer)
    # feed the first 5 as prefill, then tokens 6,7,8 one at a time
    out_prefill = model(ids[:, :5], cache=cache)
    logits_steps = [out_prefill[:, -1:, :]]
    for t in range(5, 8):
        logits_steps.append(model(ids[:, t:t + 1], cache=cache))
    stepwise_last = torch.cat(logits_steps, dim=1)   # logits at positions 4..7

    torch.testing.assert_close(full[:, 4:, :], stepwise_last, atol=1e-4, rtol=1e-4)


def test_cache_grows_by_one_per_step():
    model = _model(3)
    cache = KVCache.make(model.cfg.n_layer)
    model(torch.randint(0, 64, (1, 5)), cache=cache)
    assert cache.length == 5
    model(torch.randint(0, 64, (1, 1)), cache=cache)
    assert cache.length == 6
    # every layer's cache advances together
    assert all(lc.length == 6 for lc in cache.layers)


def test_causal_mask_blocks_future_tokens():
    """A prefix's logits must not change when later tokens are appended
    (i.e. attention is causal — no peeking ahead)."""
    model = _model(4)
    ids = torch.randint(0, 64, (1, 10))
    short = model(ids[:, :4])               # logits for first 4 positions
    long = model(ids)                        # logits for all 10
    torch.testing.assert_close(short, long[:, :4, :], atol=1e-4, rtol=1e-4)
