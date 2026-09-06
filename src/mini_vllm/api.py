"""OpenAI-compatible serving layer for the engine.

Split in two so the core is testable without a web framework:

* `handle_completion(...)` — pure function mapping an OpenAI `/v1/completions`
  request dict to a response dict, driving the continuous-batching `Engine`.
  Fully CPU-testable, no HTTP.
* `create_app(...)` — a thin FastAPI wrapper exposing `/v1/completions`,
  `/v1/models`, `/health`. FastAPI is imported lazily so this module loads
  without it; install with the `serve` extra to actually run a server.
"""

from __future__ import annotations

import time
import uuid

from .model import TinyTransformer
from .scheduler import Engine
from .tokenizer import ByteTokenizer


def handle_completion(
    model: TinyTransformer,
    tokenizer: ByteTokenizer,
    request: dict,
    model_name: str = "mini-vllm-tiny",
    max_running: int = 8,
) -> dict:
    """Serve an OpenAI `/v1/completions`-style request.

    Supports `prompt` as a string or a list of strings, and `max_tokens`. Runs
    all prompts through the continuous-batching engine and returns an
    OpenAI-shaped response (choices aligned to the input prompts, token usage).
    """
    prompt = request.get("prompt", "")
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    max_tokens = int(request.get("max_tokens", 16))
    eos_id = request.get("eos_id")  # optional; None means generate max_tokens

    engine = Engine(model, max_running=max_running, eos_id=eos_id)
    order: list[int] = []
    prompt_tok_counts: list[int] = []
    for p in prompts:
        ids = tokenizer.encode(p) or [0]     # avoid empty prompt
        prompt_tok_counts.append(len(ids))
        order.append(engine.add_request(ids, max_tokens))

    finished = engine.run()

    choices, completion_tokens = [], 0
    for idx, internal_id in enumerate(order):
        out_ids = finished[internal_id]
        completion_tokens += len(out_ids)
        finish = "stop" if (eos_id is not None and out_ids and out_ids[-1] == eos_id) else "length"
        choices.append({
            "index": idx,
            "text": tokenizer.decode(out_ids),
            "finish_reason": finish,
            "logprobs": None,
        })

    prompt_tokens = sum(prompt_tok_counts)
    return {
        "id": "cmpl-" + uuid.uuid4().hex[:24],
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_app(model: TinyTransformer, tokenizer: ByteTokenizer | None = None,
               model_name: str = "mini-vllm-tiny"):
    """Build a FastAPI app exposing the OpenAI-style endpoints. Lazy import so the
    rest of the package works without FastAPI installed (`pip install .[serve]`)."""
    from fastapi import FastAPI  # lazy

    tokenizer = tokenizer or ByteTokenizer()
    app = FastAPI(title="mini-vllm")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(request: dict):
        return handle_completion(model, tokenizer, request, model_name=model_name)

    return app
