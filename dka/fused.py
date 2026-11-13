"""
Fused DKA kernel (Triton): the one-pass update specified in dka/fused_reference.py.

Stage 1: the forward pass. One program per block of BLOCK_M queries of one (batch, head) loops over the key blocks
(up to the diagonal if causal), computes each pair's logit w = L + log E in registers, and keeps the running maximum
m and the three running sums l1 = sum p, l2 = sum p E, acc = sum p V~ (p = e^{w - m}); it returns
o1 = acc / l1, lse1 = m + log l1 and lse2 = m + log l2, the outputs of the two FlexAttention passes. q, k, v are read
in their own dtype (bf16 under autocast); the dot products accumulate in float32, and the elementwise terms are
float32. The squared norms |q_i|^2, |k_j|^2 are inputs (computed from the same q, k outside).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _dka_fwd_kernel(Q, K, V, T, QSQ, KSQ, MU, VINF, SINF, INVNU, COEF, O, LSE1, LSE2,
                    sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son, stb, H, N,
                    D2: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    CAUSAL: tl.constexpr, ROBUST: tl.constexpr, IEEE: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DV)
    row_ok = offs_m < N

    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    q_sq = tl.load(QSQ + bh * N + offs_m, mask=row_ok, other=0.0)
    t_q = tl.load(T + b * stb + offs_m, mask=row_ok, other=0.0)
    mu = tl.load(MU + h)
    v_inf = tl.load(VINF + h)
    s_inf = tl.load(SINF + h)
    inv_nu = tl.load(INVNU + h)
    coef = tl.load(COEF + h)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    l2 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)

    hi = tl.minimum((pid_m + 1) * BLOCK_M, N) if CAUSAL else N
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        col_ok = offs_n < N
        k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_v[None, :], mask=col_ok[:, None], other=0.0)
        k_sq = tl.load(KSQ + bh * N + offs_n, mask=col_ok, other=0.0)
        t_k = tl.load(T + b * stb + offs_n, mask=col_ok, other=0.0)

        if IEEE:                                                                # float32 inputs: no TF32
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))                                          # float32 accumulation
        lag = t_q[:, None] - t_k[None, :]
        lag = tl.maximum(lag, 0.0) if CAUSAL else tl.abs(lag)
        log_e = -mu * lag
        e = tl.exp(log_e)
        e2 = e * e
        sigma = s_inf - v_inf * e2
        r2 = tl.maximum(k_sq[None, :] + e2 * q_sq[:, None] - 2.0 * e * s, 0.0)
        x = r2 * inv_nu / sigma
        if ROBUST:
            phi = libdevice.log1p(x)
        else:
            phi = x
        w = log_e - coef * phi - tl.log(sigma)
        keep = col_ok[None, :] & row_ok[:, None]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        w = tl.where(keep, w, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(w, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(w - m_new[:, None])
        l1 = l1 * alpha + tl.sum(p, 1)
        l2 = l2 * alpha + tl.sum(p * e, 1)
        if IEEE:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l1[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_v[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])
    tl.store(LSE1 + bh * N + offs_m, m_i + tl.log(l1), mask=row_ok)
    tl.store(LSE2 + bh * N + offs_m, m_i + tl.log(l2), mask=row_ok)


def fused_forward(q, k, v, t, q_sq, k_sq, consts, robust, causal, block_m=64, block_n=None, num_warps=4,
                  num_stages=None):
    """
    (o1, lse1, lse2) of the fused forward pass. q, k: [B, H, N, 2D]; v: [B, H, N, Dv] (same dtype); t: [B|1, N];
    q_sq, k_sq: [B, H, N] float32; consts: (mu, v_inf, s_inf, inv_nu, coef), each [H] float32. o1 has v's dtype.
    """
    B, H, N, D2 = q.shape
    DV = v.shape[-1]
    wide = q.element_size() * max(D2, DV) > 256          # e.g. float32 at width 128: smaller key blocks, one stage
    block_n = block_n or (32 if wide else 64)            # (64 x 64 blocks need 115 KB of shared memory there)
    num_stages = num_stages or (1 if wide else 2)
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    t = t.to(torch.float32).expand(B, N).contiguous()
    q_sq, k_sq = (x.to(torch.float32).contiguous() for x in (q_sq, k_sq))
    consts = [c.to(torch.float32).contiguous() for c in consts]
    o = torch.empty(B, H, N, DV, device=q.device, dtype=v.dtype)
    lse1 = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    lse2 = torch.empty_like(lse1)
    grid = (triton.cdiv(N, block_m), B * H)
    _dka_fwd_kernel[grid](q, k, v, t, q_sq, k_sq, *consts, o, lse1, lse2,
                          q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
                          v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), t.stride(0),
                          H, N, D2=D2, DV=DV, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal, ROBUST=robust,
                          IEEE=q.dtype == torch.float32, num_warps=num_warps, num_stages=num_stages)
    return o, lse1, lse2
