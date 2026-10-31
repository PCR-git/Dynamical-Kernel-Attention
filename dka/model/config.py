"""Typed configuration of the transformer backbones (the RFA and DKA layers are configured by dka.RFAConfig and
dka.DKAConfig)."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from dka import RFAConfig, DKAConfig


@dataclass(frozen=True)
class AttentionConfig:
    """
    Baseline multi-head dot-product attention with optional positional mechanisms:

        rope:          None (NoPE) | "rope" | "sc_rope" | "xpos"   rotary embedding of queries and keys
        alibi:         linear distance bias on the logits
        decay_damping: exponential decay of the attention weights, mu_h = damping * omega_h,max
                       (applied after the softmax, as in RFA); None = no decay
        bias_damping:  linear distance bias -2 mu_h |i - j| on the logits (before the softmax), mu_h = damping *
                       omega_h,max: DKA's log h with zeta = 0, i.e. ALiBi with frequency-coupled slopes; None = none

        kerple_log:    KERPLE's logarithmic bias -p_h log(1 + a_h |i - j|) on the logits, p_h and a_h learned per head
                       (initialized uniform on [0, 2] and [0, 1], clamped at 0.01, as in the KERPLE code)
        softmax_plus_one: an extra null logit 0 in the softmax (Miller, 2023), so a head can attend to nothing

    The baselines of the paper are B1 = rope, B2 = alibi, B3 = rope + decay, B4 = sc_rope + decay,
    B5 = sc_rope + coupled linear bias, B6 = kerple_log, B7 = xpos, B8 = NoPE, B9 = rope + softmax plus one.
    """

    d_model: int
    n_heads: int
    width: int                       # total query/key/value width across heads
    rope: Optional[Literal["rope", "sc_rope", "xpos"]] = "rope"
    alibi: bool = False
    decay_damping: Optional[float] = None
    bias_damping: Optional[float] = None
    kerple_log: bool = False
    softmax_plus_one: bool = False

    def __post_init__(self):
        if self.width % self.n_heads != 0:
            raise ValueError(f"width={self.width} is not divisible by n_heads={self.n_heads}")
        if self.rope not in (None, "rope", "sc_rope", "xpos"):
            raise ValueError(f"rope must be None, 'rope', 'sc_rope' or 'xpos', got {self.rope!r}")


@dataclass(frozen=True)
class BackboneConfig:
    """
    A stack of pre-norm residual blocks, x + attn(norm1(x)) then x + ffn(norm2(x)), with a ReLU
    feedforward network, a final norm, and optional d->d projections around the stack.
    """

    attention: Union[RFAConfig, DKAConfig, AttentionConfig]
    n_blocks: int
    ffn_dim: int
    norm: Optional[Literal["layer", "rms"]] = "layer"  # None = no normalization layers at all
    attn_norm: bool = True           # normalize the block input before attention
    ffn_norm: bool = True            # normalize before the feedforward network
    io_projections: bool = True      # needed when the backbone is used directly as a regressor; redundant
                                     # inside a LanguageModel (embedding and LM head already provide them)

    def __post_init__(self):
        if self.norm not in (None, "layer", "rms"):
            raise ValueError(f"norm must be None, 'layer' or 'rms', got {self.norm!r}")

    @property
    def d_model(self):
        return self.attention.d_model
