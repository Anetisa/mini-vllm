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

## Status

Built incrementally; each layer is tested as it lands.

- [x] **KV-cache + step-by-step decoding** — cached decode proves identical to
      recomputing the full prefix; ~O(n) vs O(n²) per generation
- [x] **Paged KV-cache** — block allocator + per-sequence block tables; gather
      equals contiguous under attention, no fragmentation, sequences isolated
- [ ] **Continuous batching scheduler** — add/evict requests mid-flight
- [ ] **OpenAI-compatible endpoint** — actually serve it
- [ ] Throughput demo on GPU + write-up

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
