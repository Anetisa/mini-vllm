"""A minimal decoder-only transformer with an explicit KV-cache.

Kept deliberately small and readable — the point is to *show* how a KV-cache
works, not to be a fast or trained model. Weights are random; correctness here
means "generation with the cache produces the same tokens as without it".

Why a KV-cache: during autoregressive decoding, step t needs attention over all
previous tokens. Recomputing the whole prefix every step is O(n²) work over a
generation. Instead we cache each layer's K and V for past tokens and, per step,
compute only the new token's Q/K/V, append K/V to the cache, and attend over the
whole cache — O(n) per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    max_seq_len: int = 512


class LayerKVCache:
    """Growing K/V store for one attention layer: [B, n_head, T_so_far, head_dim]."""

    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


@dataclass
class KVCache:
    """One LayerKVCache per transformer layer."""

    layers: list[LayerKVCache] = field(default_factory=list)

    @classmethod
    def make(cls, n_layer: int) -> "KVCache":
        return cls(layers=[LayerKVCache() for _ in range(n_layer)])

    @property
    def length(self) -> int:
        return self.layers[0].length if self.layers else 0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x, layer_cache: LayerKVCache | None = None):
        B, T, D = x.shape
        past_len = layer_cache.length if layer_cache is not None else 0

        q, k, v = self.qkv(x).split(D, dim=2)
        # [B, T, D] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if layer_cache is not None:
            k, v = layer_cache.append(k, v)     # attend over full history

        # causal mask over absolute positions: query i (abs = past_len + i)
        # may attend key j (abs) iff j <= past_len + i.
        Tk = k.shape[2]
        q_pos = past_len + torch.arange(T, device=x.device)[:, None]
        k_pos = torch.arange(Tk, device=x.device)[None, :]
        mask = k_pos <= q_pos                     # [T, Tk] bool

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B,nh,T,Tk]
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v                            # [B, nh, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x, layer_cache=None):
        x = x + self.attn(self.ln1(x), layer_cache)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """GPT-style decoder with learned positional embeddings and a KV-cache path."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids, cache: KVCache | None = None):
        """input_ids: [B, T]. If `cache` is given, positions start at the cache
        length (so we can feed one token at a time during decode)."""
        B, T = input_ids.shape
        past_len = cache.length if cache is not None else 0
        pos = past_len + torch.arange(T, device=input_ids.device)

        x = self.wte(input_ids) + self.wpe(pos)[None, :, :]
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[i] if cache is not None else None
            x = block(x, layer_cache)
        x = self.ln_f(x)
        return self.lm_head(x)      # [B, T, vocab]
