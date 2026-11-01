import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dka.init import init_spectrally_coupled_rope
from ..config import AttentionConfig
from .pos_encoding import RoPE, SCRoPE, XPos, get_alibi_slopes


def softmax_plus_one(scores):
    """exp(s_j) / (1 + sum_k exp(s_k)) over the last dim: softmax with an extra null logit fixed at 0."""
    m = scores.amax(dim=-1, keepdim=True).clamp(min=0.0)
    e = torch.exp(scores - m)
    return e / (e.sum(dim=-1, keepdim=True) + torch.exp(-m))


class MultiHeadAttentionLayer(nn.Module):
    """Baseline multi-head dot-product attention (presets B1-B10); see AttentionConfig for the options."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = c = config
        self.num_heads = c.n_heads
        self.head_dim = c.width // c.n_heads

        # Projections for Q, K, V, and final output
        self.query_projection = nn.Linear(c.d_model, c.width)
        self.key_projection = nn.Linear(c.d_model, c.width)
        self.value_projection = nn.Linear(c.d_model, c.width)
        self.output_projection = nn.Linear(c.width, c.d_model)

        if c.rope == "rope":
            self.rope = RoPE(self.head_dim, yarn_original_length=c.yarn_original_length)
        elif c.rope == "sc_rope":
            self.rope = SCRoPE(self.head_dim, n_heads=c.n_heads, yarn_original_length=c.yarn_original_length)
        elif c.rope == "xpos":
            self.rope = XPos(self.head_dim)
        else:
            self.rope = None

        if c.alibi:
            self.register_buffer("alibi_slopes", get_alibi_slopes(c.n_heads))  # [1, H, 1, 1]

        if c.decay_damping is not None:
            _, mu = init_spectrally_coupled_rope(c.n_heads, self.head_dim // 2, b=c.decay_damping)
            self.register_buffer("mu", mu)

        if c.bias_damping is not None:
            _, mu = init_spectrally_coupled_rope(c.n_heads, self.head_dim // 2, b=c.bias_damping)
            self.register_buffer("bias_slopes", (2 * mu).view(1, -1, 1, 1))  # log h = -2 mu dt

        if c.kerple_log:
            self.kerple_p = nn.Parameter(2 * torch.rand(c.n_heads))
            self.kerple_a = nn.Parameter(torch.rand(c.n_heads))

    def forward(self, x, t_measure=None, causal=True, return_metadata=False):
        """
        Self-attention over regular positions. x: [B, L, d_model].
        Returns (out [B, L, d_model], {"attn_mat": attention weights [B, H, L, L]}).
        return_metadata is accepted for interface compatibility with the RFA layer (the weights are always returned).
        """
        if t_measure is not None:
            raise ValueError("Baseline attention only supports regular positions (t_measure=None)")
        c = self.config
        B, L, _ = x.shape
        Q = self.query_projection(x).view(B, L, self.num_heads, self.head_dim)
        K = self.key_projection(x).view(B, L, self.num_heads, self.head_dim)
        V = self.value_projection(x).view(B, L, self.num_heads, self.head_dim)

        if c.rope == "xpos":
            Q, K = self.rope(Q), self.rope(K, key=True)
        elif self.rope is not None:
            Q, K = self.rope(Q), self.rope(K)

        # [B, H, L, D_head]
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if c.yarn_original_length is not None and c.rope in ("rope", "sc_rope"):
            scores = scores * self.rope.yarn_logit_scale(L)

        pos = torch.arange(L, device=x.device)
        dist = (pos[:, None] - pos[None, :]).abs()  # [L, L]
        if c.alibi:
            scores = scores - self.alibi_slopes * dist
        if c.bias_damping is not None:
            scores = scores - self.bias_slopes * dist
        if c.kerple_log:
            p = self.kerple_p.clamp(min=0.01).view(1, -1, 1, 1)
            a = self.kerple_a.clamp(min=0.01).view(1, -1, 1, 1)
            scores = scores - p * torch.log1p(a * dist)
        if causal:
            scores = scores.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=x.device).tril(), float("-inf"))

        if c.softmax_plus_one:                       # an extra null logit 0: weights sum to at most 1
            attn_weights = softmax_plus_one(scores)
        else:
            attn_weights = F.softmax(scores, dim=-1)
        if c.decay_damping is not None:
            attn_weights = attn_weights * torch.exp(-self.mu.view(1, -1, 1, 1) * dist)

        context = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, L, c.width)
        return self.output_projection(context), {"attn_mat": attn_weights}
