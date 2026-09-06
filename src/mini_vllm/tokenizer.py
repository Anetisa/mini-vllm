"""A trivial byte-level tokenizer.

Our TinyTransformer has vocab_size 256, which lines up exactly with byte values,
so we can tokenize arbitrary text as its UTF-8 bytes. This lets the OpenAI-style
endpoint accept and return *text* (like a real server) without shipping a learned
tokenizer. Outputs are gibberish (the model is random) — the point is the serving
layer, not generation quality.
"""

from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")
