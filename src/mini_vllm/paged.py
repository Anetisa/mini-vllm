"""Paged KV-cache — vLLM's PagedAttention memory idea, at teaching scale.

A contiguous KV-cache forces each sequence's keys/values to live in one growing
tensor. With many concurrent sequences of unknown length that means either
reserving max_len per sequence (huge waste) or copying to grow — fragmentation
either way.

Paging borrows from OS virtual memory: cut the cache into fixed-size **blocks**
(pages), each holding `block_size` tokens. A per-sequence **block table** maps
logical token positions to physical blocks, which need not be contiguous.
Allocating is popping a free block from a pool; freeing returns it. No external
fragmentation; the only waste is the last, partially-filled block per sequence.

This module is pure memory-management logic — fully testable on CPU. The
correctness property: gathering a sequence's K/V from its blocks reproduces the
contiguous cache exactly, so attention is unchanged.
"""

from __future__ import annotations

import torch


class BlockAllocator:
    """A pool of fixed-size physical blocks; hand them out and take them back."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._used: set[int] = set()

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("out of KV-cache blocks")
        b = self._free.pop()
        self._used.add(b)
        return b

    def free(self, block: int) -> None:
        if block not in self._used:
            raise ValueError(f"freeing block {block} that isn't allocated")
        self._used.remove(block)
        self._free.append(block)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)


class PagedKVCache:
    """Block-paged K/V storage for a whole model.

    Physical pools are per layer, shape [num_blocks, n_head, block_size, head_dim].
    A block table (per sequence, shared across layers) maps logical block index
    -> physical block id. Token i lives in logical block i // block_size at slot
    i % block_size.
    """

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = head_dim
        self.block_size = block_size
        self.allocator = BlockAllocator(num_blocks)
        self.k_pool = [
            torch.zeros(num_blocks, n_head, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layer)
        ]
        self.v_pool = [
            torch.zeros(num_blocks, n_head, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(n_layer)
        ]
        self.block_tables: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    # ------------------------------------------------------------------ #
    # sequence lifecycle                                                 #
    # ------------------------------------------------------------------ #
    def add_sequence(self, seq_id: int) -> None:
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} already exists")
        self.block_tables[seq_id] = []
        self.lengths[seq_id] = 0

    def free_sequence(self, seq_id: int) -> None:
        for b in self.block_tables[seq_id]:
            self.allocator.free(b)
        del self.block_tables[seq_id]
        del self.lengths[seq_id]

    # ------------------------------------------------------------------ #
    # write / read                                                       #
    # ------------------------------------------------------------------ #
    def reserve(self, seq_id: int, n_new: int) -> None:
        """Grow a sequence by `n_new` tokens, allocating blocks as needed."""
        new_len = self.lengths[seq_id] + n_new
        blocks_needed = (new_len + self.block_size - 1) // self.block_size
        table = self.block_tables[seq_id]
        while len(table) < blocks_needed:
            table.append(self.allocator.allocate())
        self.lengths[seq_id] = new_len

    def write(self, seq_id: int, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write the last k.shape[1] tokens' K/V for one layer.

        k, v: [n_head, T_new, head_dim]. Assumes `reserve` already advanced the
        length to include these tokens.
        """
        T_new = k.shape[1]
        length = self.lengths[seq_id]
        table = self.block_tables[seq_id]
        start = length - T_new
        for i in range(T_new):
            pos = start + i
            blk = table[pos // self.block_size]
            slot = pos % self.block_size
            self.k_pool[layer][blk, :, slot, :] = k[:, i, :]
            self.v_pool[layer][blk, :, slot, :] = v[:, i, :]

    def gather(self, seq_id: int, layer: int):
        """Reconstruct contiguous K, V for a sequence: [n_head, length, head_dim]."""
        length = self.lengths[seq_id]
        table = self.block_tables[seq_id]
        ks, vs = [], []
        for pos in range(length):
            blk = table[pos // self.block_size]
            slot = pos % self.block_size
            ks.append(self.k_pool[layer][blk, :, slot, :])
            vs.append(self.v_pool[layer][blk, :, slot, :])
        k = torch.stack(ks, dim=1)   # [n_head, length, head_dim]
        v = torch.stack(vs, dim=1)
        return k, v

    def num_blocks_for(self, seq_id: int) -> int:
        return len(self.block_tables[seq_id])
