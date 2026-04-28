import pytest

from nanovlm.train.data.preference import PreferenceLoader


class _StubTokenizer:
    eos_token_id = 0

    def encode(self, text):
        return [hash(t) & 0xFF for t in str(text).split() if t]


def test_preference_loader_masks_only_response_tokens():
    records = iter([{"prompt": "two plus two", "chosen": "four", "rejected": "five"}])
    loader = PreferenceLoader(records, _StubTokenizer())
    batch = next(loader)
    # prompt_ids 3, chosen_response 1 → chosen_ids length 4, mask [0,0,0,1]
    assert batch["chosen_ids"].shape == (1, 4)
    assert batch["chosen_mask"].tolist() == [[0, 0, 0, 1]]
    # rejected response 1 → rejected mask [0,0,0,1]
    assert batch["rejected_ids"].shape == (1, 4)
    assert batch["rejected_mask"].tolist() == [[0, 0, 0, 1]]


def test_chat_loader_assistant_only_mask():
    from nanovlm.train.data.chat import ChatLoader, build_chat_record

    rec = build_chat_record({"prompt": "hi", "response": "hello"})
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][1]["role"] == "assistant"

    loader = ChatLoader(iter([{"prompt": "hi", "response": "hello"}]), _StubTokenizer(), seq_len=64)
    out = next(loader)
    assert out["input_ids"].shape == (1, 64)
    assert int(out["loss_mask"].sum()) > 0
    assert int(out["loss_mask"].sum()) < 64


def test_rlvr_group_sampler_repeats_record():
    from nanovlm.train.data.rlvr import GroupSampler, RlvrRecordIter

    recs = RlvrRecordIter([{"question": "q1", "answer": "a"}, {"question": "q2", "answer": "b"}])
    sampler = GroupSampler(recs, group_size=4)
    g1 = next(sampler)
    g2 = next(sampler)
    assert len(g1) == 4 and all(r["question"] == "q1" for r in g1)
    assert len(g2) == 4 and all(r["question"] == "q2" for r in g2)


def test_yaml_subset_parser_handles_mixture_config(tmp_path):
    from nanovlm.train.data.mixture import load_mixture_config

    p = tmp_path / "mix.yaml"
    p.write_text(
        "name: test_mix\n"
        "seq_len: 1024\n"
        "batch_size: 2\n"
        "sources:\n"
        "  - name: a\n"
        "    weight: 0.7\n"
        "    format: parquet\n"
        "    glob: a/*.parquet\n"
        "  - name: b\n"
        "    weight: 0.3\n"
        "    format: jsonl\n"
        "    glob: b/*.jsonl\n"
    )
    cfg = load_mixture_config(p)
    assert cfg["name"] == "test_mix"
    assert cfg["seq_len"] == 1024
    assert cfg["batch_size"] == 2
    assert len(cfg["sources"]) == 2
    assert cfg["sources"][0]["name"] == "a"
    assert cfg["sources"][0]["weight"] == 0.7
    assert cfg["sources"][1]["format"] == "jsonl"
