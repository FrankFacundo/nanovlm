from pathlib import Path

from .config import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from .cache import HybridCache
from .layers import RMSNorm, RMSNormGated, SwiGLUMLP
from .rotary import TextRotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .attention import Qwen3_5Attention
from .linear_attention import (
    Qwen3_5GatedDeltaNet,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
    l2norm,
)
from .decoder import Qwen3_5DecoderLayer, Qwen3_5TextModel, Qwen3_5ForCausalLM
from .vision import Qwen3_5VisionModel
from .model import Qwen3_5Model, Qwen3_5ForConditionalGeneration
from .weights import load_qwen3_5_weights
from .tokenizer import Qwen2Tokenizer
from .image_processor import Qwen2VLImageProcessor, smart_resize
from .chat_template import (
    AssistantTurn,
    ToolCall,
    detect_pending_tool_call,
    parse_assistant,
    render_chat,
    render_chat_for_training,
    render_tool_response,
)

DEFAULT_MODEL_PATH = Path.home() / "Models" / "Qwen" / "Qwen3.5-0.8B"

__all__ = [
    "DEFAULT_MODEL_PATH",
    "Qwen3_5Config",
    "Qwen3_5TextConfig",
    "Qwen3_5VisionConfig",
    "HybridCache",
    "RMSNorm",
    "RMSNormGated",
    "SwiGLUMLP",
    "TextRotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    "Qwen3_5Attention",
    "Qwen3_5GatedDeltaNet",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
    "l2norm",
    "Qwen3_5DecoderLayer",
    "Qwen3_5TextModel",
    "Qwen3_5ForCausalLM",
    "Qwen3_5VisionModel",
    "Qwen3_5Model",
    "Qwen3_5ForConditionalGeneration",
    "load_qwen3_5_weights",
    "Qwen2Tokenizer",
    "Qwen2VLImageProcessor",
    "smart_resize",
    "render_chat",
    "render_chat_for_training",
    "parse_assistant",
    "detect_pending_tool_call",
    "render_tool_response",
    "ToolCall",
    "AssistantTurn",
]
