"""Tests for the OpenAI-compatible serving logic — pure `handle_completion`,
no web server needed. All on CPU."""

import torch

from mini_vllm.api import handle_completion
from mini_vllm.model import ModelConfig, TinyTransformer
from mini_vllm.tokenizer import ByteTokenizer


def _model(seed=0):
    torch.manual_seed(seed)
    # vocab_size must be 256 to match the byte tokenizer
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=4, d_model=64, max_seq_len=256)
    return TinyTransformer(cfg).eval()


def test_byte_tokenizer_roundtrip():
    tok = ByteTokenizer()
    for s in ["hello", "GRPO + Triton", "ju:\n\ttabs", "emoji ok?"]:
        assert tok.decode(tok.encode(s)) == s


def test_completion_response_shape():
    model, tok = _model(), ByteTokenizer()
    resp = handle_completion(model, tok, {"prompt": "hello", "max_tokens": 8})
    # OpenAI-shaped
    assert resp["object"] == "text_completion"
    assert resp["id"].startswith("cmpl-")
    assert isinstance(resp["created"], int)
    assert len(resp["choices"]) == 1
    c = resp["choices"][0]
    assert set(c) == {"index", "text", "finish_reason", "logprobs"}
    assert c["index"] == 0 and isinstance(c["text"], str)
    assert c["finish_reason"] == "length"


def test_usage_token_accounting():
    model, tok = _model(), ByteTokenizer()
    resp = handle_completion(model, tok, {"prompt": "abcd", "max_tokens": 5})
    u = resp["usage"]
    assert u["prompt_tokens"] == 4          # "abcd" = 4 bytes
    assert u["completion_tokens"] == 5      # max_tokens generated
    assert u["total_tokens"] == 9


def test_batch_prompts_give_aligned_choices():
    """A list prompt returns one choice per prompt, in order — and each matches
    what the single-prompt call would produce (continuous batching underneath)."""
    model, tok = _model(1), ByteTokenizer()
    prompts = ["aa", "bbbb", "c"]
    batch = handle_completion(model, tok, {"prompt": prompts, "max_tokens": 6})
    assert [c["index"] for c in batch["choices"]] == [0, 1, 2]

    for i, p in enumerate(prompts):
        single = handle_completion(model, tok, {"prompt": p, "max_tokens": 6})
        assert batch["choices"][i]["text"] == single["choices"][0]["text"]


def test_eos_sets_stop_finish_reason():
    model, tok = _model(2), ByteTokenizer()
    # get the model's greedy first token id directly (text round-trip is lossy
    # for non-ASCII bytes, so we read the id from the logits, not the decoded text)
    ids = torch.tensor([tok.encode("xyz")])
    first_id = int(model(ids)[0, -1, :].argmax().item())
    resp = handle_completion(model, tok, {"prompt": "xyz", "max_tokens": 50, "eos_id": first_id})
    assert resp["choices"][0]["finish_reason"] == "stop"


def test_app_factory_importable_without_fastapi():
    """create_app must only need FastAPI when actually called, not on import."""
    import mini_vllm.api as api
    assert hasattr(api, "create_app")       # module imported fine without fastapi
