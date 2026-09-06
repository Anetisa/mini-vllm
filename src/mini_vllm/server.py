"""Run the OpenAI-compatible server: `python -m mini_vllm.server`.

Needs the `serve` extra (fastapi + uvicorn). Serves a tiny *random* model, so
completions are gibberish — this demonstrates the serving layer end to end
(POST /v1/completions with an OpenAI-shaped body), not text quality.
"""

from __future__ import annotations


def main():
    import argparse
    import uvicorn

    from .api import create_app
    from .model import ModelConfig, TinyTransformer
    from .tokenizer import ByteTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    args = p.parse_args()

    cfg = ModelConfig(vocab_size=256, n_layer=args.n_layer, d_model=args.d_model)
    model = TinyTransformer(cfg).eval()
    app = create_app(model, ByteTokenizer())
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
