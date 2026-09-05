"""Tests for the paged KV-cache — allocator accounting, round-trip fidelity,
and equivalence to a contiguous cache under attention. All CPU."""

import torch

from mini_vllm.paged import BlockAllocator, PagedKVCache


# --------------------------------------------------------------------------- #
# Block allocator                                                            #
# --------------------------------------------------------------------------- #
def test_allocator_accounting():
    a = BlockAllocator(4)
    assert a.num_free == 4 and a.num_used == 0
    b0, b1 = a.allocate(), a.allocate()
    assert a.num_free == 2 and a.num_used == 2
    assert b0 != b1                       # never hand out the same block twice
    a.free(b0)
    assert a.num_free == 3 and a.num_used == 1


def test_allocator_raises_when_exhausted():
    a = BlockAllocator(2)
    a.allocate(); a.allocate()
    try:
        a.allocate()
        assert False, "should have raised on exhaustion"
    except RuntimeError:
        pass


def test_allocator_reuses_freed_blocks_no_fragmentation():
    """Free then re-allocate must succeed by reusing blocks (no leak/fragment)."""
    a = BlockAllocator(3)
    blocks = [a.allocate() for _ in range(3)]
    for b in blocks:
        a.free(b)
    assert a.num_free == 3
    # can allocate all 3 again
    again = [a.allocate() for _ in range(3)]
    assert sorted(again) == sorted(blocks)


# --------------------------------------------------------------------------- #
# Paged cache round-trip                                                     #
# --------------------------------------------------------------------------- #
def _cache(num_blocks=16, block_size=4, n_layer=2, n_head=3, head_dim=8):
    return PagedKVCache(n_layer, n_head, head_dim, num_blocks, block_size)


def test_gather_reconstructs_written_kv():
    """Writing K/V through the block table and gathering back is lossless."""
    torch.manual_seed(0)
    cache = _cache(block_size=4)
    cache.add_sequence(0)
    T = 10                                 # spans 3 blocks (4+4+2)
    k = torch.randn(3, T, 8)
    v = torch.randn(3, T, 8)
    cache.reserve(0, T)
    cache.write(0, layer=0, k=k, v=v)
    gk, gv = cache.gather(0, layer=0)
    torch.testing.assert_close(gk, k, atol=0, rtol=0)
    torch.testing.assert_close(gv, v, atol=0, rtol=0)


def test_incremental_append_matches_one_shot():
    """Appending one token at a time (decode) == writing all at once (prefill)."""
    torch.manual_seed(1)
    T = 9
    k = torch.randn(3, T, 8)
    v = torch.randn(3, T, 8)

    one = _cache(block_size=4); one.add_sequence(0)
    one.reserve(0, T); one.write(0, 0, k, v)
    gk1, gv1 = one.gather(0, 0)

    inc = _cache(block_size=4); inc.add_sequence(0)
    for t in range(T):
        inc.reserve(0, 1)
        inc.write(0, 0, k[:, t:t + 1, :], v[:, t:t + 1, :])
    gk2, gv2 = inc.gather(0, 0)

    torch.testing.assert_close(gk1, gk2, atol=0, rtol=0)
    torch.testing.assert_close(gv1, gv2, atol=0, rtol=0)


def test_paged_attention_equals_contiguous():
    """The point: attention over the paged (gathered) K/V equals attention over
    a plain contiguous K/V. Paging changes storage, not math."""
    torch.manual_seed(2)
    n_head, T, hd = 4, 11, 16
    q = torch.randn(n_head, 1, hd)          # a single decode query
    k = torch.randn(n_head, T, hd)
    v = torch.randn(n_head, T, hd)

    def attn(q, k, v):
        s = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        return torch.softmax(s, dim=-1) @ v

    contiguous = attn(q, k, v)

    cache = _cache(block_size=4, n_head=n_head, head_dim=hd)
    cache.add_sequence(0)
    cache.reserve(0, T); cache.write(0, 0, k, v)
    gk, gv = cache.gather(0, 0)
    paged = attn(q, gk, gv)

    torch.testing.assert_close(paged, contiguous, atol=1e-6, rtol=1e-6)


def test_free_returns_blocks_to_pool():
    cache = _cache(num_blocks=8, block_size=4)
    cache.add_sequence(0)
    cache.reserve(0, 10)                    # uses 3 blocks
    assert cache.allocator.num_used == 3
    cache.free_sequence(0)
    assert cache.allocator.num_used == 0 and cache.allocator.num_free == 8


def test_sequences_are_isolated():
    """Two sequences must not share blocks or corrupt each other's data."""
    torch.manual_seed(3)
    cache = _cache(block_size=4)
    cache.add_sequence(0); cache.add_sequence(1)
    k0 = torch.randn(3, 5, 8); v0 = torch.randn(3, 5, 8)
    k1 = torch.randn(3, 7, 8); v1 = torch.randn(3, 7, 8)
    cache.reserve(0, 5); cache.write(0, 0, k0, v0)
    cache.reserve(1, 7); cache.write(1, 0, k1, v1)
    # block tables must be disjoint
    assert set(cache.block_tables[0]).isdisjoint(cache.block_tables[1])
    gk0, _ = cache.gather(0, 0)
    gk1, _ = cache.gather(1, 0)
    torch.testing.assert_close(gk0, k0, atol=0, rtol=0)
    torch.testing.assert_close(gk1, k1, atol=0, rtol=0)
