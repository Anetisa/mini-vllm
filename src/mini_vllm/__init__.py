"""mini-vllm: an educational LLM inference engine — KV-cache, paged memory,
continuous batching — built from scratch and tested on CPU."""

from .model import KVCache, LayerKVCache, ModelConfig, TinyTransformer
from .generate import generate_cached, generate_no_cache
from .paged import BlockAllocator, PagedKVCache
from .scheduler import Engine, Request, Scheduler, State

__all__ = [
    "ModelConfig",
    "TinyTransformer",
    "KVCache",
    "LayerKVCache",
    "generate_cached",
    "generate_no_cache",
    "BlockAllocator",
    "PagedKVCache",
    "Engine",
    "Scheduler",
    "Request",
    "State",
]

__version__ = "0.2.0"
