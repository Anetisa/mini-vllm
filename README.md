# mini-vllm

**An educational LLM inference engine, built from scratch — KV-cache, paged memory, and continuous batching — with every piece validated on CPU.**

Serving LLMs efficiently is a systems problem, not a modeling one. The ideas that
make engines like vLLM fast — caching past keys/values, managing that cache in
fixed-size *pages*, and *continuously batching* requests that arrive and finish
at different times — are what this repo implements and explains, on a tiny
transformer you can run on a laptop.

> Companion to [triton-llm-kernels](https://github.com/Anetisa/triton-llm-kernels)
> and [grpo-countdown](https://github.com/Anetisa/grpo-countdown). Same approach:
> each component is small, readable, and covered by CPU tests — the GPU is only
> for the final throughput demo.

**How it fits together:** see the [architecture write-up](docs/architecture.md).

## Status

Built incrementally; each layer is tested as it lands.

- [x] **KV-cache + step-by-step decoding** — cached decode proves identical to
      recomputing the full prefix; ~O(n) vs O(n²) per generation
- [x] **Paged KV-cache** — block allocator + per-sequence block tables; gather
      equals contiguous under attention, no fragmentation, sequences isolated
- [x] **Continuous batching scheduler** — iteration-level admit/retire; batched
      output proven identical to isolated generation; capacity + EOS handling
- [x] **OpenAI-compatible endpoint** — `/v1/completions` over the engine; pure
      request→response logic tested on CPU, thin FastAPI wrapper
- [x] Architecture [write-up](docs/architecture.md)
- [ ] Throughput demo on GPU (fused batched PagedAttention step)

## The idea so far: KV-cache

During autoregressive decoding, step *t* needs attention over all previous
tokens. Recomputing the whole prefix every step is **O(n²)** work across a
generation. A KV-cache stores each layer's K and V for past tokens, so per step
we compute only the new token's Q/K/V, append K/V to the cache, and attend over
the whole cache — **O(n) per step**.

The correctness property is simple and testable: **caching is a speed
optimization, so cached decoding must produce exactly the same tokens as the
uncached reference.** That equivalence is the core test.

```python
import torch
from mini_vllm import TinyTransformer, ModelConfig, generate_cached, generate_no_cache

model = TinyTransformer(ModelConfig()).eval()
prompt = torch.randint(0, 256, (1, 8))

a = generate_no_cache(model, prompt, max_new_tokens=32)   # O(n^2) reference
b = generate_cached(model, prompt, max_new_tokens=32)     # O(n) with KV-cache
assert torch.equal(a, b)                                  # identical tokens
```

On CPU the cache already pulls ahead as the sequence grows (the gap widens with
length, as O(n) vs O(n² predicts):

```
gen  32 tokens | no-cache 0.154s | cached 0.062s | speedup 2.5x
gen  64 tokens | no-cache 0.381s | cached 0.133s | speedup 2.9x
gen 128 tokens | no-cache 1.129s | cached 0.313s | speedup 3.6x
```

## Serve (OpenAI-compatible)

The engine sits behind a standard `/v1/completions` endpoint. The request/response
logic is a pure function (`handle_completion`) tested without any web server; a
thin FastAPI layer exposes it.

```bash
pip install -e ".[serve]"
python -m mini_vllm.server            # serves a tiny random model on :8000

curl localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"prompt": "hello", "max_tokens": 16}'
```

The model is random, so the text is gibberish — this demonstrates the serving
path (batched via the continuous-batching engine, OpenAI-shaped response with
token usage), not generation quality. In code, without a server:

```python
from mini_vllm import TinyTransformer, ModelConfig, ByteTokenizer, handle_completion
model = TinyTransformer(ModelConfig(vocab_size=256)).eval()
resp = handle_completion(model, ByteTokenizer(),
                         {"prompt": ["hi", "there"], "max_tokens": 8})
# resp["choices"] -> one per prompt; resp["usage"] -> token counts
```

## Install & test

```bash
pip install -e ".[dev]"
make test        # all on CPU — no GPU needed for the logic
```

## References

- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention*
  (vLLM, 2023)

## License

MIT — see [LICENSE](LICENSE).
