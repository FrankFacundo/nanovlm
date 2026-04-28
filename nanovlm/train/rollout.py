"""Batched on-policy rollouts and multi-turn tool-use rollouts.

``group_rollout`` is the primary helper for GRPO/DAPO: given a single prompt
text, it generates ``group_size`` completions (text + per-token logprobs) in
a single batched call. The caller supplies a verifier to score each rollout.

``tool_use_rollout`` interleaves model generation with tool calls. After each
generate step, it parses the trailing text for a complete
``<tool_call>...</tool_call>`` block; if found, it executes the tool, appends
``<tool_response>{json}</tool_response>`` to the prompt, and continues. The
loop terminates on EOS or after ``max_turns`` turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from nanovlm.models.qwen3_5.chat_template import (
    IM_END,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    parse_assistant,
)

from .engine import generate, sample_next


@dataclass
class GroupRollout:
    prompt_text: str
    prompt_ids: torch.Tensor                 # [1, T_prompt]
    response_ids: list[torch.Tensor]         # [G] each [T_resp_i]
    response_text: list[str]
    response_logprobs: list[torch.Tensor]    # [G] each [T_resp_i]


@torch.no_grad()
def group_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    *,
    group_size: int,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device | str = "cpu",
) -> GroupRollout:
    """Generate ``group_size`` rollouts of ``prompt_text``.

    Returns a ``GroupRollout`` whose ``response_ids`` and ``response_logprobs``
    contain only the *new* tokens (no prompt prefix).
    """
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
    batched = prompt_ids.expand(group_size, -1).contiguous()
    out = generate(
        model, batched,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_logprobs=True,
        use_cache=True,
    )
    new_tokens = out.sequences[:, prompt_ids.size(1):]
    new_logp = out.new_token_logprobs  # [G, T_new] (may include EOS-pad)

    # Strip post-EOS padding per row
    eos_set = set()
    if eos_token_id is not None:
        eos_set = set([eos_token_id]) if isinstance(eos_token_id, int) else set(eos_token_id)

    response_ids = []
    response_lp = []
    response_text = []
    for i in range(group_size):
        toks = new_tokens[i].tolist()
        cut = len(toks)
        for j, t in enumerate(toks):
            if t in eos_set:
                cut = j + 1  # include EOS
                break
        response_ids.append(new_tokens[i, :cut])
        if new_logp is not None:
            response_lp.append(new_logp[i, :cut])
        else:
            response_lp.append(torch.zeros(cut, device=device))
        response_text.append(tokenizer.decode(toks[:cut]))

    return GroupRollout(
        prompt_text=prompt_text,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_text=response_text,
        response_logprobs=response_lp,
    )


def stack_padded(seqs: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of 1-D tensors to the longest; return (stacked, mask)."""
    if not seqs:
        return torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long)
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype, device=seqs[0].device)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.long, device=seqs[0].device)
    for i, s in enumerate(seqs):
        n = s.size(0)
        out[i, :n] = s
        mask[i, :n] = 1
    return out, mask


@dataclass
class ToolUseStep:
    role: str               # "assistant" or "tool"
    text: str
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_result: Any = None
    tool_error: str | None = None


@dataclass
class ToolUseTrajectory:
    prompt_text: str
    steps: list[ToolUseStep] = field(default_factory=list)
    final_text: str = ""
    success: bool = False
    reward: float = 0.0


@torch.no_grad()
def tool_use_rollout(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    tools: dict[str, Callable[[dict], Any]],
    *,
    max_turns: int = 6,
    max_new_tokens_per_turn: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    eos_token_id: int | list[int] | None = None,
    device: torch.device | str = "cpu",
) -> ToolUseTrajectory:
    """Interleave model generation with tool execution until EOS or ``max_turns``.

    ``tools`` maps tool name → callable that takes a dict of arguments and
    returns a JSON-serializable result. This implementation re-encodes the
    full conversation each turn (no KV-cache reuse across turns) for
    correctness with the hybrid (full + DeltaNet) architecture; the caller is
    expected to keep ``max_turns`` modest.
    """
    traj = ToolUseTrajectory(prompt_text=prompt_text)
    convo = prompt_text
    for _ in range(max_turns):
        ids = torch.tensor([tokenizer.encode(convo)], dtype=torch.long, device=device)
        out = generate(
            model, ids,
            max_new_tokens=max_new_tokens_per_turn,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=False,
            use_cache=True,
        )
        new_tokens = out.sequences[0, ids.size(1):].tolist()
        new_text = tokenizer.decode(new_tokens)
        turn = parse_assistant(new_text)
        if turn.tool_calls:
            for tc in turn.tool_calls:
                step = ToolUseStep(role="assistant", text=new_text, tool_name=tc.name, tool_args=tc.arguments)
                fn = tools.get(tc.name)
                if fn is None:
                    step.tool_error = f"unknown tool: {tc.name!r}"
                    convo += new_text + IM_END + "\n"
                    convo += f"<|im_start|>user\n{TOOL_RESPONSE_OPEN}\n{{\"error\": \"{step.tool_error}\"}}\n{TOOL_RESPONSE_CLOSE}{IM_END}\n<|im_start|>assistant\n"
                else:
                    try:
                        result = fn(tc.arguments or {})
                        step.tool_result = result
                    except Exception as e:
                        step.tool_error = str(e)
                        result = {"error": step.tool_error}
                    import json as _json
                    convo += new_text
                    if not new_text.rstrip().endswith(IM_END):
                        convo += IM_END + "\n"
                    convo += f"<|im_start|>user\n{TOOL_RESPONSE_OPEN}\n{_json.dumps(result, default=str)[:4096]}\n{TOOL_RESPONSE_CLOSE}{IM_END}\n<|im_start|>assistant\n"
                traj.steps.append(step)
            continue
        # No tool call: terminal assistant message
        traj.steps.append(ToolUseStep(role="assistant", text=new_text))
        traj.final_text = turn.content
        break
    return traj
