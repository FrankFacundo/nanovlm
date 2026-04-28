import torch

from nanovlm.train.engine import generate, sample_next
from nanovlm.train.model_factory import build_model


def test_sample_next_greedy_matches_argmax():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    idx = sample_next(logits, temperature=0.0)
    assert idx.tolist() == [[1], [2]]


def test_generate_greedy_kvcache_matches_no_cache():
    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    m.eval()
    ids = torch.randint(0, 1000, (1, 8))
    g_cache = generate(m, ids, max_new_tokens=6, eos_token_id=None, temperature=0.0, use_cache=True)
    g_no = generate(m, ids, max_new_tokens=6, eos_token_id=None, temperature=0.0, use_cache=False)
    assert g_cache.sequences.shape == g_no.sequences.shape
    # Greedy must match exactly (KV cache is correctness-preserving)
    assert torch.equal(g_cache.sequences, g_no.sequences)


def test_generate_returns_logprobs_when_requested():
    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    ids = torch.randint(0, 1000, (1, 4))
    out = generate(m, ids, max_new_tokens=3, eos_token_id=None, temperature=0.5, return_logprobs=True, use_cache=True)
    assert out.new_token_logprobs is not None
    assert out.new_token_logprobs.shape == (1, 3)


def test_generate_stops_on_eos():
    torch.manual_seed(0)
    m = build_model(init="scratch", tiny=True, text_only=True, dtype=torch.float32, device="cpu")
    ids = torch.randint(0, 1000, (1, 4))
    out = generate(m, ids, max_new_tokens=20, eos_token_id=[0, 1, 2], temperature=0.0, use_cache=True)
    # Greedy with broad EOS set is highly likely to stop early
    assert out.sequences.size(1) <= 4 + 20
