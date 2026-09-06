"""Tests for the continuous-batching scheduler + engine (CPU).

Key property: mixing requests that arrive/finish at different times must not
change any individual result vs. generating it alone.
"""

import torch

from mini_vllm.generate import generate_cached
from mini_vllm.model import ModelConfig, TinyTransformer
from mini_vllm.scheduler import Engine, Request, Scheduler, State


def _model(seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, n_layer=3, n_head=4, d_model=64, max_seq_len=256)
    return TinyTransformer(cfg).eval()


def _alone(model, prompt_ids, n):
    """Greedy tokens generated for a prompt on its own (reference)."""
    out = generate_cached(model, torch.tensor([prompt_ids]), max_new_tokens=n)
    return out[0, len(prompt_ids):].tolist()


# --------------------------------------------------------------------------- #
# Correctness: batched == isolated                                           #
# --------------------------------------------------------------------------- #
def test_continuous_batching_matches_isolated():
    model = _model()
    prompts = {
        0: ([1, 2, 3], 12),
        1: ([9, 8], 20),
        2: ([4, 4, 4, 4], 7),
        3: ([15, 2, 30], 15),
    }
    ref = {rid: _alone(model, p, n) for rid, (p, n) in prompts.items()}

    # capacity smaller than #requests so slots must recycle mid-flight
    eng = Engine(model, max_running=2)
    ids = {eng.add_request(p, n): rid for rid, (p, n) in prompts.items()}
    out = eng.run()

    for internal_id, rid in ids.items():
        assert out[internal_id] == ref[rid], f"request {rid} diverged under batching"


def test_never_exceeds_capacity():
    model = _model(1)
    eng = Engine(model, max_running=2)
    for _ in range(6):
        eng.add_request([1, 2, 3], 10)
    while eng.scheduler.has_work:
        eng.step()
        assert len(eng.scheduler.running) <= 2
    assert eng.max_running_seen <= 2
    assert len(eng.finished) == 6           # all completed


def test_waiting_requests_get_admitted_as_slots_free():
    model = _model(2)
    eng = Engine(model, max_running=1)       # strictly one at a time
    a = eng.add_request([1, 2], 5)
    b = eng.add_request([3, 4], 5)
    eng.step()                               # only 'a' runs, 'b' waits
    assert len(eng.scheduler.running) == 1
    assert eng.scheduler.waiting and eng.scheduler.waiting[0].req_id == b
    out = eng.run()
    assert set(out.keys()) == {a, b} and all(len(v) == 5 for v in out.values())


def test_max_new_tokens_respected():
    model = _model(3)
    eng = Engine(model, max_running=4)
    rid = eng.add_request([7, 7], 9)
    out = eng.run()
    assert len(out[rid]) == 9


def test_eos_stops_generation_early():
    """If a token equals eos, the request finishes before max_new_tokens."""
    model = _model(4)
    prompt = [5, 6, 7]
    first = _alone(model, prompt, 1)[0]      # the model's greedy first token
    eng = Engine(model, max_running=2, eos_id=first)
    rid = eng.add_request(prompt, max_new_tokens=50)
    out = eng.run()
    assert out[rid][-1] == first
    assert len(out[rid]) < 50                # stopped early on eos


def test_scheduler_queue_mechanics():
    """Unit-test the scheduler independent of the model."""
    s = Scheduler(max_running=2)
    reqs = [Request(i, [1], 3) for i in range(4)]
    for r in reqs:
        s.add(r)
    assert len(s.waiting) == 4 and len(s.running) == 0
    s.admit()
    assert len(s.running) == 2 and len(s.waiting) == 2
    assert all(r.state == State.RUNNING for r in s.running)
    s.running[0].output_ids = [1, 2, 3]      # mark one finished (hit max_new_tokens)
    done = s.retire_finished()
    assert len(done) == 1 and done[0].state == State.FINISHED
    s.admit()                                # freed slot -> admit a waiter
    assert len(s.running) == 2 and len(s.waiting) == 1
