# mini-vllm architecture

This engine is four small layers stacked into an LLM server. Each solves one
problem and is tested in isolation on CPU; together they reproduce the ideas that
make production engines like vLLM fast. This doc explains how they fit.

```
            ┌──────────────────────────────────────────────┐
  HTTP  ───▶│  /v1/completions  (api.py, server.py)         │   serving layer
            │  text ──ByteTokenizer──▶ token ids            │
            └───────────────────────┬──────────────────────┘
                                     │ add_request / run
            ┌────────────────────────▼─────────────────────┐
            │  Engine + Scheduler (scheduler.py)            │   iteration-level
            │  waiting queue ⇄ running set (bounded)        │   scheduling
            │  each step(): admit → advance all → retire    │   (continuous
            └───────────────┬───────────────────────────────┘    batching)
                            │ one token / running request / step
            ┌───────────────▼───────────────────────────────┐
            │  TinyTransformer (model.py)                    │   compute
            │  prefill (whole prompt) then 1-token decode    │
            └───────────────┬───────────────────────────────┘
                            │ read past K/V, append new K/V
            ┌───────────────▼───────────────────────────────┐
            │  KV-cache (model.py) / Paged KV-cache (paged)  │   memory
            │  O(n) decode; blocks + block tables, no frag   │
            └────────────────────────────────────────────────┘
```

## The request lifecycle

1. A client POSTs an OpenAI-shaped body to `/v1/completions`. `handle_completion`
   tokenizes each prompt to byte ids and hands them to the `Engine`.
2. The `Scheduler` puts each request in a **waiting** queue. Every `step()` it
   **admits** requests into a bounded **running** set (capacity = how many can
   decode at once).
3. Each `step()` advances every running request by exactly one token: the first
   touch **prefills** the whole prompt (populating that request's KV-cache); after
   that, each step is a **single-token decode** that reads history from the cache.
4. A request that hits `max_tokens` or emits EOS is **retired**, freeing its slot;
   a waiting request is admitted in its place — the running set stays full.
5. When nothing is waiting or running, `run()` returns each request's tokens,
   which `handle_completion` detokenizes into an OpenAI response with usage.

## Why each layer exists

**KV-cache — the memory that makes decoding O(n).** In autoregressive decoding,
step *t* attends over all previous tokens. Recomputing the whole prefix each step
is O(n²) over a generation. The cache stores each layer's past K and V, so a step
computes only the new token's Q/K/V, appends K/V, and attends over the cache —
O(n) per step. Correctness is pinned by an equivalence test: cached decoding
produces the *same tokens* as the uncached reference.

**Paged KV-cache — memory management without fragmentation.** A contiguous cache
per sequence forces you to either over-reserve (waste) or copy to grow
(fragment). Paging cuts the cache into fixed-size blocks and maps each sequence's
token positions to physical blocks via a per-sequence **block table**, exactly
like OS virtual memory. Allocation is popping a free block; freeing returns it;
the only waste is the last partial block. The equivalence test: attention over
the gathered (paged) K/V equals attention over a contiguous K/V — paging changes
storage, not math.

**Continuous batching — keep the running set full.** Static batching waits for a
whole batch to finish before starting new work, so the batch drains as sequences
finish at different lengths and the GPU idles. The scheduler instead admits and
retires requests *between steps* (iteration-level scheduling). Correctness test:
interleaving requests that start and finish at different times gives each request
the *same output* as generating it alone.

**Serving layer — an OpenAI-shaped door.** The engine sits behind
`/v1/completions`. The request→response mapping is a pure function
(`handle_completion`) tested without a web server; FastAPI is a thin, lazily
imported wrapper. A byte tokenizer (matching the model's 256 vocab) lets it speak
text like a real server.

## Testing philosophy

Every layer is validated on CPU with tiny tensors and a tiny random model —
25 tests covering: cached==uncached decode, causal masking, allocator accounting,
paged==contiguous attention, sequence isolation, batched==isolated generation,
capacity/EOS handling, and OpenAI response shape/usage. A GPU is only needed for
a throughput demonstration, not for correctness. This is the same "build and
prove on CPU, spend GPU only at the end" approach used across the companion
repos.

## Scope and roadmap

The engine is intentionally readable over maximally fast. The honest boundary:

- **Execution is a per-request loop, not a fused kernel.** Each `step()` advances
  running requests one at a time. The *scheduling* (the vLLM contribution) is
  fully implemented; fusing a step's running set into a single batched
  **PagedAttention** kernel — attending over block tables directly, no
  gather/pad — is the throughput optimization. That kernel is exactly the kind of
  thing in the companion [triton-llm-kernels](https://github.com/Anetisa/triton-llm-kernels)
  repo, and wiring it in is the natural next step.
- The paged cache is built and tested, but the default engine path uses a simple
  per-request cache for clarity; making paging the shared execution backend is a
  small integration on top.
- Natural extensions: a `/v1/chat/completions` endpoint with a chat template,
  sampling (temperature/top-p) beyond greedy, prefix/block sharing across
  requests, and preemption/eviction under memory pressure.

None of these change the ideas above — they're performance and feature layers on
the same skeleton.
