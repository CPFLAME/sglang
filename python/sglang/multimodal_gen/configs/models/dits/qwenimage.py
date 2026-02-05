# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class QwenImageArchConfig(DiTArchConfig):
    patch_size: int = 1
    in_channels: int = 64
    out_channels: int | None = None
    num_layers: int = 19
    num_single_layers: int = 38
    attention_head_dim: int = 128
    num_attention_heads: int = 24
    joint_attention_dim: int = 4096
    pooled_projection_dim: int = 768
    guidance_embeds: bool = False
    axes_dims_rope: Tuple[int, int, int] = (16, 56, 56)
    zero_cond_t: bool = False

    stacked_params_mapping: list[tuple[str, str, str]] = field(default_factory=list)

    param_names_mapping: dict = field(
        default_factory=lambda: {
            # LoRA mappings
            r"^(transformer_blocks\.\d+\.attn\..*\.lora_[AB])\.default$": r"\1",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class QwenImageEditPlus_2511_ArchConfig(DiTArchConfig):
    zero_cond_t: bool = True

    def __post_init__(self):
        # NOTE:
        # For Qwen-Image-Edit-2511, many architecture fields are injected from HF configs
        # via `ModelConfig.update_model_arch()`. Known fields (e.g. num_attention_heads)
        # are dataclass fields, while others (e.g. attention_head_dim, in_channels,
        # out_channels, patch_size, num_layers, joint_attention_dim) are stored in
        # ArchConfig.extra_attrs.
        #
        # The base DiTArchConfig does NOT derive hidden_size / num_channels_latents,
        # so hidden_size may stay 0, which later breaks attention backend selection
        # (e.g. head_size = hidden_size // num_heads -> 0).
        super().__post_init__()

        # Derive hidden_size if possible.
        try:
            head_dim = getattr(self, "attention_head_dim", None)
        except Exception:
            head_dim = None
        if self.hidden_size == 0 and self.num_attention_heads and head_dim:
            self.hidden_size = int(self.num_attention_heads) * int(head_dim)

        # Derive num_channels_latents from (out_channels or in_channels) if present.
        if self.num_channels_latents == 0:
            out_channels = getattr(self, "out_channels", None)
            in_channels = getattr(self, "in_channels", None)
            ch = out_channels if out_channels is not None else in_channels
            if ch is not None:
                self.num_channels_latents = int(ch)


@dataclass
class QwenImageDitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=QwenImageArchConfig)

    prefix: str = "qwenimage"


@dataclass
class QwenImageEditPlus_2511_DitConfig(DiTConfig):
    arch_config: DiTArchConfig = field(
        default_factory=QwenImageEditPlus_2511_ArchConfig
    )

    prefix: str = "qwenimageedit"
