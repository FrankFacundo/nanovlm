"""Pure-Python chat template for Qwen3.5.

Replaces the Jinja2 ``apply_chat_template`` for hot training paths. The
grammar matches the markers shipped in Qwen3.5's ``tokenizer_config.json``:

  - role boundaries:   ``<|im_start|>{role}\\n{content}<|im_end|>\\n``
  - thinking:          ``<think>...</think>``
  - tool call:         ``<tool_call>{json}</tool_call>``
  - tool response:     ``<tool_response>{json}</tool_response>``
  - vision wrap:       ``<|vision_start|><|image_pad|><|vision_end|>``
  - EOS:               ``<|endoftext|>``

Provides:
  - ``render_chat`` — render a list of messages to a single string.
  - ``render_chat_for_training`` — render and return token ids + an assistant-only
    loss mask (for SFT).
  - ``parse_assistant`` — parse a generated assistant turn into
    ``{think, tool_calls, content}``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS = "<|endoftext|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

DEFAULT_SYSTEM = "You are Qwen, a helpful assistant."

_TOOL_CALL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"\s*(.*?)\s*" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)
_THINK_RE = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL,
)


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class AssistantTurn:
    content: str
    think: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


def _content_to_text(content: Any) -> str:
    """Flatten Qwen3.5 multimodal message content into a string.

    Image items emit ``<|vision_start|><|image_pad|><|vision_end|>`` placeholders;
    the image processor and ``Qwen3_5Model`` later replace ``<|image_pad|>`` with
    the right number of patch tokens via ``masked_scatter``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _item_to_text(content)
    if isinstance(content, list):
        return "".join(_item_to_text(item) for item in content)
    return str(content)


def _item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return str(item)
    t = item.get("type")
    if t == "text" or "text" in item:
        return str(item.get("text", ""))
    if t == "image" or "image" in item or "image_url" in item:
        return f"{VISION_START}{IMAGE_PAD}{VISION_END}"
    if t == "video" or "video" in item:
        return f"{VISION_START}{VIDEO_PAD}{VISION_END}"
    return ""


def _render_tool_call(tc: dict | ToolCall) -> str:
    if isinstance(tc, ToolCall):
        payload = tc.to_dict()
    else:
        if "function" in tc:
            fn = tc["function"]
            payload = {
                "name": fn.get("name"),
                "arguments": fn.get("arguments", {}),
            }
        else:
            payload = {
                "name": tc.get("name"),
                "arguments": tc.get("arguments", {}),
            }
        if isinstance(payload["arguments"], str):
            try:
                payload["arguments"] = json.loads(payload["arguments"])
            except json.JSONDecodeError:
                pass
    return f"{TOOL_CALL_OPEN}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOL_CALL_CLOSE}"


def _render_assistant_body(message: dict, *, enable_thinking: bool) -> str:
    pieces: list[str] = []
    think = message.get("thinking") or message.get("think")
    if think and enable_thinking:
        pieces.append(f"{THINK_OPEN}\n{think.strip()}\n{THINK_CLOSE}\n")
    elif enable_thinking and message.get("content") and not message.get("tool_calls"):
        pass

    content = _content_to_text(message.get("content"))
    if content:
        pieces.append(content)

    for tc in message.get("tool_calls", []) or []:
        pieces.append(_render_tool_call(tc))
    return "".join(pieces)


def render_chat(
    messages: list[dict],
    *,
    add_generation_prompt: bool = False,
    enable_thinking: bool = True,
    default_system: str | None = DEFAULT_SYSTEM,
) -> str:
    """Render a chat into the Qwen3.5 wire format.

    Each message is a dict with at least ``role``. Roles supported:
      - ``system``: text content.
      - ``user``: text content or list of items (text/image/video).
      - ``assistant``: text content, optional ``thinking`` field, optional
        ``tool_calls`` list (OpenAI-style or ``{name, arguments}``).
      - ``tool``: text content rendered as a ``user`` turn wrapped in
        ``<tool_response>...</tool_response>``.
    """
    out: list[str] = []
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and default_system:
        out.append(f"{IM_START}system\n{default_system}{IM_END}\n")

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = _content_to_text(msg.get("content", ""))
            out.append(f"{IM_START}system\n{content}{IM_END}\n")
        elif role == "user":
            content = _content_to_text(msg.get("content", ""))
            out.append(f"{IM_START}user\n{content}{IM_END}\n")
        elif role == "assistant":
            body = _render_assistant_body(msg, enable_thinking=enable_thinking)
            out.append(f"{IM_START}assistant\n{body}{IM_END}\n")
        elif role == "tool":
            content = _content_to_text(msg.get("content", ""))
            out.append(
                f"{IM_START}user\n{TOOL_RESPONSE_OPEN}\n{content}\n{TOOL_RESPONSE_CLOSE}{IM_END}\n"
            )
        else:
            raise ValueError(f"unsupported chat role: {role!r}")

    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


def render_chat_for_training(
    messages: list[dict],
    tokenizer,
    *,
    enable_thinking: bool = True,
    default_system: str | None = DEFAULT_SYSTEM,
    seq_len: int | None = None,
) -> dict:
    """Render messages and return token ids plus an assistant-only loss mask.

    The mask is 1 on tokens emitted from assistant turns (excluding the
    ``<|im_start|>assistant\\n`` prefix but including the closing ``<|im_end|>``)
    and 0 elsewhere. This is the standard SFT supervision pattern.

    Returns:
        ``{"input_ids": list[int], "loss_mask": list[int], "text": str}``.
        If ``seq_len`` is set, both lists are right-padded with ``<|endoftext|>``
        (mask 0) or truncated to length ``seq_len``.
    """
    input_ids: list[int] = []
    loss_mask: list[int] = []
    rendered_parts: list[str] = []

    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system and default_system:
        prefix = f"{IM_START}system\n{default_system}{IM_END}\n"
        ids = tokenizer.encode(prefix)
        input_ids.extend(ids)
        loss_mask.extend([0] * len(ids))
        rendered_parts.append(prefix)

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            head = f"{IM_START}assistant\n"
            body = _render_assistant_body(msg, enable_thinking=enable_thinking)
            tail = f"{IM_END}\n"

            head_ids = tokenizer.encode(head)
            body_ids = tokenizer.encode(body)
            tail_ids = tokenizer.encode(tail)

            input_ids.extend(head_ids)
            loss_mask.extend([0] * len(head_ids))
            input_ids.extend(body_ids)
            loss_mask.extend([1] * len(body_ids))
            input_ids.extend(tail_ids)
            loss_mask.extend([1] * len(tail_ids))
            rendered_parts.append(head + body + tail)
        else:
            text = render_chat(
                [msg],
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
                default_system=None,
            )
            ids = tokenizer.encode(text)
            input_ids.extend(ids)
            loss_mask.extend([0] * len(ids))
            rendered_parts.append(text)

    if seq_len is not None:
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            loss_mask = loss_mask[:seq_len]
        else:
            pad_id = _eos_id(tokenizer)
            pad_n = seq_len - len(input_ids)
            input_ids.extend([pad_id] * pad_n)
            loss_mask.extend([0] * pad_n)

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "text": "".join(rendered_parts),
    }


def _eos_id(tokenizer) -> int:
    eid = getattr(tokenizer, "eos_token_id", None)
    if eid is not None:
        return int(eid)
    added = getattr(tokenizer, "_added", {})
    if EOS in added:
        return int(added[EOS])
    raise ValueError("tokenizer has no eos token")


def parse_assistant(text: str) -> AssistantTurn:
    """Parse a single generated assistant turn into structured fields.

    Strips any leading ``<|im_start|>assistant\\n`` and trailing ``<|im_end|>``,
    extracts a ``<think>...</think>`` block if present, extracts all
    ``<tool_call>...</tool_call>`` blocks (parsed as JSON when possible), and
    returns the remaining textual content.
    """
    s = text
    if s.startswith(f"{IM_START}assistant\n"):
        s = s[len(f"{IM_START}assistant\n") :]
    end = s.find(IM_END)
    if end != -1:
        s = s[:end]

    think = None
    m = _THINK_RE.search(s)
    if m:
        think = m.group(1).strip()
        s = s[: m.start()] + s[m.end() :]

    tool_calls: list[ToolCall] = []
    cleaned: list[str] = []
    last = 0
    for m in _TOOL_CALL_RE.finditer(s):
        cleaned.append(s[last : m.start()])
        last = m.end()
        raw = m.group(1).strip()
        try:
            payload = json.loads(raw)
            name = str(payload.get("name", ""))
            args = payload.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(ToolCall(name=name, arguments=args, raw=raw))
        except json.JSONDecodeError:
            tool_calls.append(ToolCall(name="", arguments={}, raw=raw))
    cleaned.append(s[last:])
    content = "".join(cleaned).strip()
    return AssistantTurn(content=content, think=think, tool_calls=tool_calls)


def render_tool_response(tool_name: str, result: Any) -> dict:
    """Build a chat message for a tool response, ready to feed back to ``render_chat``."""
    if not isinstance(result, str):
        try:
            result = json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            result = str(result)
    return {"role": "tool", "name": tool_name, "content": result}


def detect_pending_tool_call(generated_text: str) -> bool:
    """Return True iff ``generated_text`` ends with a complete ``<tool_call>...</tool_call>``."""
    last_open = generated_text.rfind(TOOL_CALL_OPEN)
    last_close = generated_text.rfind(TOOL_CALL_CLOSE)
    return last_open != -1 and last_close > last_open
