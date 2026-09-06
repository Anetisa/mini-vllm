"""mini-vllm: an educational LLM inference engine — KV-cache, paged memory,
continuous batching — built from scratch and tested on CPU."""

from .model import KVCache, LayerKVCache, ModelConfig, TinyTransformer
from .generate import generate_cached, generate_no_cache
from .paged import BlockAllocator, PagedKVCache
from .scheduler import Engine, Request, Scheduler, State
from .tokenizer import ByteTokenizer
from .api import handle_completion, create_app

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
    "ByteTokenizer",
    "handle_completion",
    "create_app",
]

__version__ = "0.3.0"
