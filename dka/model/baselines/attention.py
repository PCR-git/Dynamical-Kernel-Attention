import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from dka.flex import biased_attention
from dka.init import init_spectrally_coupled_rope
from ..config import AttentionConfig
from .pos_encoding import RoPE, SCRoPE, XPos, get_alibi_slopes


def softmax_plus_one(scores, null_logit=0.0):
    """exp(s_j) / (exp(b) + sum_k exp(s_k)) over the last dim: softmax with an extra null logit b (default 0)."""
    b = null_logit.to(scores.dtype) if torch.is_tensor(null_logit) else scores.new_tensor(null_logit)
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), b)
    e = torch.exp(scores - m)
    return e / (e.sum(dim=-1, keepdim=True) + torch.exp(b - m))


# scaled_dot_product_attention's backends in order of preference. PyTorch's own choice falls back to the
# memory-efficient kernel where flash attention is not built (e.g. compute capability 12.0, RTX 50xx), although
# cuDNN's is 2-3x faster there; math is the reference (CPU, float64)
_SDPA_PRIORITY = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                  SDPBackend.MATH]


class MultiHeadAttentionLayer(nn.Module):
    """Baseline multi-head dot-product attention (presets B1-B10); see AttentionConfig for the options."""

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = c = config
        self.query_chunk = None       # evaluation memory: process the queries in blocks of this size (None = all)
        self.num_heads = c.n_heads
        self.head_dim = c.width // c.n_heads

        # Projections for Q, K, V, and final output
        self.query_projection = nn.Linear(c.d_model, c.width)
        self.key_projection = nn.Linear(c.d_model, c.width)
        self.value_projection = nn.Linear(c.d_model, c.width)
        self.output_projection = nn.Linear(c.width, c.d_model)

        if c.rope == "rope":
            self.rope = RoPE(self.head_dim, base=c.rope_base, yarn_original_length=c.yarn_original_length)
        elif c.rope == "sc_rope":
            self.rope = SCRoPE(self.head_dim, n_heads=c.n_heads, base=c.rope_base,
                               yarn_original_length=c.yarn_original_length)
        elif c.rope == "xpos":
            self.rope = XPos(self.head_dim)
        else:
            self.rope = None

        if c.alibi:
            self.register_buffer("alibi_slopes", get_alibi_slopes(c.n_heads))  # [1, H, 1, 1]

        if c.decay_damping is not None:
            _, mu = init_spectrally_coupled_rope(c.n_heads, self.head_dim // 2, b=c.decay_damping, base=c.rope_base)
            self.register_buffer("mu", mu)

        if c.bias_damping is not None:
            _, mu = init_spectrally_coupled_rope(c.n_heads, self.head_dim // 2, b=c.bias_damping, base=c.rope_base)
            self.register_buffer("bias_slopes", (2 * c.bias_slope_scale * mu).view(1, -1, 1, 1))  # log h = -2 mu dt

        if c.kerple_log:
            self.kerple_p = nn.Parameter(2 * torch.rand(c.n_heads))
            self.kerple_a = nn.Parameter(torch.rand(c.n_heads))

        if c.learned_null:
            self.null_logit = nn.Parameter(torch.zeros(1, c.n_heads, 1, 1))   # b_h: the null slot holds V_i

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
        if c.kernel == "fused":
            context = self._fused(Q, K, V, causal).transpose(1, 2).reshape(B, L, c.width)
            return self.output_projection(context), {"attn_mat": None}
        chunk = self.query_chunk if self.query_chunk and L > self.query_chunk else L
        rows = [self._rows(Q, K, V, i0, min(i0 + chunk, L), causal) for i0 in range(0, L, chunk)]
        context = torch.cat([r[0] for r in rows], dim=2).transpose(1, 2).reshape(B, L, c.width)
        attn_weights = rows[0][1] if len(rows) == 1 else None   # the full matrix only without chunking
        return self.output_projection(context), {"attn_mat": attn_weights}

    def _fused(self, Q, K, V, causal):
        """kernel='fused': the same attention without materializing the weights (Q, K, V: [B, H, L, D_head])."""
        c = self.config
        L = K.shape[2]
        scale = 1 / math.sqrt(self.head_dim)
        if c.yarn_original_length is not None and c.rope in ("rope", "sc_rope"):
            scale = scale * self.rope.yarn_logit_scale(L)
        slopes = None
        if c.alibi:
            slopes = self.alibi_slopes.view(-1)
        if c.bias_damping is not None:
            slopes = self.bias_slopes.view(-1) if slopes is None else slopes + self.bias_slopes.view(-1)
        null = 0.0 if c.softmax_plus_one else (self.null_logit.view(1, -1, 1) if c.learned_null else None)
        if slopes is None and null is None:
            with sdpa_kernel(_SDPA_PRIORITY, set_priority=True):
                context = F.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)
            mass = 1.0
        else:
            dtype = V.dtype
            f32 = torch.promote_types(dtype, torch.float32)       # the bias and null slot in at least float32
            q, k, v = (X.to(dtype).contiguous() for X in (Q, K, V))
            zero = torch.zeros(self.num_heads, device=Q.device)
            context, lse = biased_attention(q, k, v, (zero if slopes is None else slopes).to(f32).contiguous(),
                                            scale, causal)
            mass = 1.0
            if null is not None:                     # weight on the keys: Z / (Z + e^b), Z = e^lse
                mass = torch.sigmoid(lse.to(f32) - null)[..., None]
                context = (context.to(f32) * mass).to(dtype)
        if c.output == "innovation":                 # sum_j w_j (V_j - V_i)
            context = context - mass * V
        return context

    def _rows(self, Q, K, V, i0, i1, causal):
        """Attention of queries i0..i1-1 over all keys (Q, K, V: [B, H, L, D_head]): (context rows, weights)."""
        c = self.config
        L = K.shape[2]
        scores = torch.matmul(Q[:, :, i0:i1], K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if c.yarn_original_length is not None and c.rope in ("rope", "sc_rope"):
            scores = scores * self.rope.yarn_logit_scale(L)

        pos = torch.arange(L, device=Q.device)
        dist = (pos[i0:i1, None] - pos[None, :]).abs()  # [nq, L]
        if c.alibi:
            scores = scores - self.alibi_slopes * dist
        if c.bias_damping is not None:
            scores = scores - self.bias_slopes * dist
        if c.kerple_log:
            p = self.kerple_p.clamp(min=0.01).view(1, -1, 1, 1)
            a = self.kerple_a.clamp(min=0.01).view(1, -1, 1, 1)
            scores = scores - p * torch.log1p(a * dist)
        if causal:
            scores = scores.masked_fill(pos[None, :] > pos[i0:i1, None], float("-inf"))

        if c.softmax_plus_one:                       # an extra null logit 0: weights sum to at most 1
            attn_weights = softmax_plus_one(scores)
        elif c.learned_null:                         # null logit b_h, whose value is V_i (enters below)
            attn_weights = softmax_plus_one(scores, self.null_logit)
        else:
            attn_weights = F.softmax(scores, dim=-1)
        if c.decay_damping is not None:
            attn_weights = attn_weights * torch.exp(-self.mu.view(1, -1, 1, 1) * dist)
        context = torch.matmul(attn_weights, V)
        if c.output == "innovation":                 # sum_j w_j V_j + w_null V_i - V_i = sum_j w_j (V_j - V_i)
            context = context - attn_weights.sum(dim=-1, keepdim=True) * V[:, :, i0:i1]
        return context, attn_weights
