"""
Model constructors that take the notebook-style `args` namespace (paper notebooks, presets, tests).

Each translates `args` into typed configs (RFAConfig / DKAConfig / AttentionConfig / BackboneConfig) once,
at construction, and keeps the call signature and return format of the original class. New code should
build Backbone(BackboneConfig(...)) directly. With args.attention_type = "dka", the RFA constructors build
DKA layers instead (same args for the dynamics and noise).
"""

import torch.nn as nn

from dka import RFAConfig, DKAConfig
from .blocks import Block, build_attention
from .config import AttentionConfig, BackboneConfig
from .baselines import MultiHeadAttentionLayer
from .model_utils import resolve_multihead_dims
from .networks import Backbone

_NORM_NAMES = {None: None, nn.LayerNorm: "layer", nn.RMSNorm: "rms"}


def _norm_name(Norm):
    if Norm not in _NORM_NAMES:
        raise ValueError(f"Norm must be nn.LayerNorm, nn.RMSNorm or None, got {Norm}")
    return _NORM_NAMES[Norm]


def attention_config_from_args(args, d_model, n_heads, width):
    """Baseline selection from the notebook flags (B1 = use_rope, B2 = use_alibi, B3/B4 = use_relative_decay_vanilla,
    B5 = use_sc_linear_bias, B6 = use_kerple_log, B7 = use_xpos, B8 = no flags (NoPE), B9 = use_softmax_one, B10 = use_sc_linear_bias + use_softmax_one)."""
    bias_damping = None
    if getattr(args, "use_sc_linear_bias", False):
        rope = "sc_rope"
        damping = None
        bias_damping = args.damping
    elif getattr(args, "use_relative_decay_vanilla", False):
        rope = "sc_rope" if args.use_SC_RoPE else "rope"
        damping = args.damping
    else:
        rope = "xpos" if getattr(args, "use_xpos", False) else ("rope" if args.use_rope else None)
        damping = None
    return AttentionConfig(d_model=d_model, n_heads=n_heads, width=width, rope=rope, alibi=args.use_alibi,
                           decay_damping=damping, bias_damping=bias_damping,
                           kerple_log=getattr(args, "use_kerple_log", False),
                           softmax_plus_one=getattr(args, "use_softmax_one", False),
                           yarn_original_length=getattr(args, "yarn_original_length", None),
                           rope_base=getattr(args, "rope_base", 10000.0),
                           bias_slope_scale=getattr(args, "bias_slope_scale", 1.0),
                           output=getattr(args, "baseline_output", "estimate"),
                           learned_null=getattr(args, "baseline_null", False),
                           kernel="fused" if getattr(args, "dka_kernel", "explicit") in ("flex", "triton")
                           else "explicit")


def rfa_config_from_args(args, n_heads, input_dim, query_key_dim=None, value_dim=None,
                         query_key_dim_total=None, value_dim_total=None):
    if query_key_dim is None or value_dim is None:
        query_key_dim, value_dim, _, _ = resolve_multihead_dims(n_heads, query_key_dim, value_dim,
                                                                query_key_dim_total, value_dim_total)
    if query_key_dim != value_dim:
        raise ValueError(f"RFA requires equal query/key and value head dims, got {query_key_dim} and {value_dim}")
    attention_type = getattr(args, "attention_type", "rfa")
    if attention_type not in ("rfa", "dka"):
        raise ValueError(f"args.attention_type must be 'rfa' or 'dka', got {attention_type!r}")
    Config = DKAConfig if attention_type == "dka" else RFAConfig
    return Config.from_args(args, d_model=input_dim, n_heads=n_heads, head_dim=query_key_dim)


def rfa_backbone_config(args, attention, n_blocks, hidden_dim, Norm, io_projections=True):
    """BackboneConfig of an RFA network from the notebook flags for the real-valued norms and residual."""
    if not getattr(args, "use_outer_residual", True):
        raise ValueError("args.use_outer_residual=False is no longer supported (it discarded the attention output).")
    return BackboneConfig(attention=attention, n_blocks=n_blocks, ffn_dim=hidden_dim, norm=_norm_name(Norm),
                          attn_norm=args.use_real_input_norm, ffn_norm=args.use_real_output_norm,
                          io_projections=io_projections)


def build_rfa_layer(args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
    """Build a MultiheadIsotropicRFA (or, with args.attention_type = "dka", DKA) layer from `args` and head dimensions."""
    return build_attention(rfa_config_from_args(args, n_heads, input_dim, query_key_dim, value_dim,
                                                query_key_dim_total, value_dim_total))


##########################################################################################
# Single attention layers (no residuals, norms or FFN)
##########################################################################################

class SelfAttentionBlock(nn.Module):
    """A single baseline attention layer. forward returns (out, attention weights [B, H, L, L])."""

    def __init__(self, input_dim, qkv_dim, num_heads, args):
        super().__init__()
        self.attn = MultiHeadAttentionLayer(attention_config_from_args(args, input_dim, num_heads, qkv_dim))

    def forward(self, x, t_measure=None, causal=True):
        out, info = self.attn(x, causal=causal)  # t_measure is ignored, as in the original
        return out, info["attn_mat"]


class RFA_Block(nn.Module):
    """A single RFA layer. Diagnostics are returned in eval mode when args.compute_metadata is True."""

    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None, value_dim_total=None):
        super().__init__()
        # (ModuleList kept so that checkpoint keys stay 'layers.0.*')
        self.layers = nn.ModuleList([build_rfa_layer(args, n_heads, input_dim, query_key_dim, value_dim,
                                                     query_key_dim_total, value_dim_total)])
        self.args = args

    def forward(self, inputs, t_measure=None, causal=True):
        return_metadata = not self.training and getattr(self.args, "compute_metadata", False)
        return self.layers[0](inputs, t_measure=t_measure, causal=causal, return_metadata=return_metadata)


##########################################################################################
# Blocks
##########################################################################################

class TransformerBlock(Block):
    """Baseline block. forward(x, causal) returns (out, attention weights)."""

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, Norm=nn.RMSNorm):
        attention = attention_config_from_args(args, input_dim, num_heads, qkv_dim)
        super().__init__(BackboneConfig(attention=attention, n_blocks=1, ffn_dim=hidden_dim, norm=_norm_name(Norm)))

    def forward(self, x, causal=True):
        out, info = super().forward(x, causal=causal)
        return out, info["attn_mat"]


class RFATransformerBlock(Block):
    """RFA block. forward(x, t_measure, causal, return_metadata) returns (out, info dict)."""

    def __init__(self, args, n_heads, input_dim, query_key_dim=None, value_dim=None, query_key_dim_total=None,
                 value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm):
        attention = rfa_config_from_args(args, n_heads, input_dim, query_key_dim, value_dim,
                                         query_key_dim_total, value_dim_total)
        super().__init__(rfa_backbone_config(args, attention, 1, hidden_dim, Norm))


##########################################################################################
# Networks
##########################################################################################

class TransformerNetwork(Backbone):
    """Baseline network. forward(x, causal) returns (out, list of each block's attention weights)."""

    def __init__(self, input_dim, qkv_dim, hidden_dim, num_heads, args, num_blocks=3, Norm=nn.RMSNorm, io_projections=True):
        attention = attention_config_from_args(args, input_dim, num_heads, qkv_dim)
        config = BackboneConfig(attention=attention, n_blocks=num_blocks, ffn_dim=hidden_dim, norm=_norm_name(Norm),
                                io_projections=io_projections)
        super().__init__(config, grad_checkpointing=getattr(args, "grad_checkpointing", False))
        self.args = args

    def forward(self, x, causal=True, **kwargs):
        out, infos = super().forward(x, causal=causal)
        return out, [info["attn_mat"] for info in infos]


class RFATransformerNetwork(Backbone):
    """
    RFA network. forward(x, t_measure, causal) returns (out, info dict of the last block); diagnostics
    are included in eval mode when args.compute_metadata is True (read at every call).
    """

    def __init__(self, args, num_blocks=2, n_heads=1, input_dim=None, query_key_dim=None, value_dim=None,
                 query_key_dim_total=None, value_dim_total=None, hidden_dim=None, Norm=nn.RMSNorm, io_projections=True):
        attention = rfa_config_from_args(args, n_heads, input_dim, query_key_dim, value_dim,
                                         query_key_dim_total, value_dim_total)
        config = rfa_backbone_config(args, attention, num_blocks, hidden_dim, Norm, io_projections)
        super().__init__(config, grad_checkpointing=getattr(args, "grad_checkpointing", False))
        self.args = args

    def forward(self, x, t_measure=None, causal=True):
        return_metadata = not self.training and getattr(self.args, "compute_metadata", False)
        out, infos = super().forward(x, t_measure=t_measure, causal=causal, return_metadata=return_metadata)
        return out, infos[-1]
