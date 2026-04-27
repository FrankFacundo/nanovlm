from nanovlm.models.qwen3_5.chat_template import (
    IM_END,
    IM_START,
    THINK_CLOSE,
    THINK_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    detect_pending_tool_call,
    parse_assistant,
    render_chat,
)


def test_render_chat_basic_roles():
    text = render_chat(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
        add_generation_prompt=False,
        default_system=None,
    )
    assert text == (
        f"{IM_START}system\nBe concise.{IM_END}\n"
        f"{IM_START}user\nping{IM_END}\n"
        f"{IM_START}assistant\npong{IM_END}\n"
    )


def test_render_chat_default_system_and_generation_prompt():
    text = render_chat([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert text.startswith(f"{IM_START}system\n")
    assert text.endswith(f"{IM_START}assistant\n")


def test_render_chat_thinking_inline():
    text = render_chat(
        [{"role": "assistant", "content": "Final.", "thinking": "consider X"}],
        default_system=None,
    )
    assert THINK_OPEN in text and THINK_CLOSE in text
    assert "consider X" in text and "Final." in text


def test_render_chat_multimodal_user():
    text = render_chat(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]}],
        default_system=None,
    )
    assert "<|vision_start|><|image_pad|><|vision_end|>" in text
    assert "what?" in text


def test_render_chat_tool_call_payload():
    text = render_chat(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "calc", "arguments": {"expr": "2+2"}}],
            }
        ],
        default_system=None,
    )
    assert TOOL_CALL_OPEN in text and TOOL_CALL_CLOSE in text
    assert '"name": "calc"' in text


def test_parse_assistant_extracts_think_and_tool_calls():
    raw = (
        f"{IM_START}assistant\n"
        f"{THINK_OPEN}\nstep 1\n{THINK_CLOSE}\n"
        "answer prefix"
        f"{TOOL_CALL_OPEN}\n"
        '{"name": "calc", "arguments": {"expr": "2+2"}}\n'
        f"{TOOL_CALL_CLOSE}"
        " trailing"
        f"{IM_END}"
    )
    turn = parse_assistant(raw)
    assert turn.think == "step 1"
    assert turn.content == "answer prefix trailing"
    assert len(turn.tool_calls) == 1
    tc = turn.tool_calls[0]
    assert tc.name == "calc"
    assert tc.arguments == {"expr": "2+2"}


def test_detect_pending_tool_call():
    assert not detect_pending_tool_call("hello")
    assert not detect_pending_tool_call(f"hello {TOOL_CALL_OPEN} {{")
    assert detect_pending_tool_call(f"hello {TOOL_CALL_OPEN} {{}} {TOOL_CALL_CLOSE}")


class _StubTokenizer:
    """Whitespace tokenizer for round-trip tests; eos_token_id is required."""
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [hash(t) & 0xFFFF for t in text.split() if t]


def test_render_chat_for_training_assistant_only_mask():
    from nanovlm.models.qwen3_5.chat_template import render_chat_for_training

    out = render_chat_for_training(
        [
            {"role": "user", "content": "ping"},
            {"role": "assistant", "content": "pong"},
        ],
        _StubTokenizer(),
        default_system=None,
    )
    assert len(out["input_ids"]) == len(out["loss_mask"])
    assert sum(out["loss_mask"]) > 0
    assert sum(out["loss_mask"]) < len(out["loss_mask"])
