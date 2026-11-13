"""
Fused DKA kernel (Triton): the one-pass update specified in dka/fused_reference.py.

Forward pass. One program per block of BLOCK_M queries of one (batch, head) loops over the key blocks (up to the
diagonal if causal), computes each pair's logit w = L + log E in registers, and keeps the running maximum m and the
three running sums l1 = sum p, l2 = sum p E, acc = sum p V~ (p = e^{w - m}); it returns o1 = acc / l1,
lse1 = m + log l1 and lse2 = m + log l2, the outputs of the two FlexAttention passes. q, k, v are read in their own
dtype (bf16 under autocast); the dot products accumulate in float32, and the elementwise terms are float32. The
squared norms |q_i|^2, |k_j|^2 are inputs (computed from the same q, k outside).

Causal attention runs the key blocks below the diagonal without masks and only the diagonal blocks with them. The
softmax exponentials are base 2 (the running maximum is kept in units of log2). Block sizes, warps and pipeline
stages are tuned per shape on first use (_CONFIGS; those that do not fit in shared memory are skipped).
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.runtime.errors import OutOfResources


@triton.jit
def _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base, v_base, kn_base, t_base,
              skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
              D2: tl.constexpr, DV: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
              MASK: tl.constexpr, ROBUST: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr = False):
    """One key block: the pair logits and the update of the running maximum and sums. MASK: causal and column masks
    (diagonal blocks, and every block if not causal)."""
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DV)
    if MASK:
        col_ok = offs_n < N
        k = tl.load(K + k_base + offs_n[:, None] * skn + offs_d[None, :], mask=col_ok[:, None], other=0.0)
        v = tl.load(V + v_base + offs_n[:, None] * svn + offs_v[None, :], mask=col_ok[:, None], other=0.0)
        k_sq = tl.load(KSQ + kn_base + offs_n, mask=col_ok, other=0.0)
        t_k = tl.load(T + t_base + offs_n, mask=col_ok, other=0.0)
    else:
        k = tl.load(K + k_base + offs_n[:, None] * skn + offs_d[None, :])
        v = tl.load(V + v_base + offs_n[:, None] * svn + offs_v[None, :])
        k_sq = tl.load(KSQ + kn_base + offs_n)
        t_k = tl.load(T + t_base + offs_n)

    if IEEE:                                                                    # float32 inputs: no TF32
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k))                                              # float32 accumulation
    if PLAIN:                         # benchmarking only: softmax attention, the same kernel without DKA's terms
        w = s * coef * 1.4426950408889634
        e = w
    else:
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
        w = (log_e - coef * phi - tl.log(sigma)) * 1.4426950408889634          # in units of log2
    if MASK:
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        w = tl.where(keep, w, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(w, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(w - m_new[:, None])
    l1 = l1 * alpha + tl.sum(p, 1)
    if not PLAIN:
        l2 = l2 * alpha + tl.sum(p * e, 1)
    if IEEE:
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
    else:
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, l1, l2, m_new


@triton.jit
def _dka_fwd_kernel(Q, K, V, T, QSQ, KSQ, MU, VINF, SINF, INVNU, COEF, O, LSE1, LSE2,
                    sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sob, soh, son, stb, H, N,
                    D2: tl.constexpr, DV: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    CAUSAL: tl.constexpr, ROBUST: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr):
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
    k_base, v_base, kn_base, t_base = b * skb + h * skh, b * svb + h * svh, bh * N, b * stb

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l1 = tl.zeros([BLOCK_M], dtype=tl.float32)
    l2 = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)

    if CAUSAL:
        diag = pid_m * BLOCK_M                       # key blocks before the diagonal: every pair visible, no masks
        for start_n in range(0, diag, BLOCK_N):
            acc, l1, l2, m_i = _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base,
                                         v_base, kn_base, t_base, skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
                                         D2, DV, BLOCK_N, CAUSAL, False, ROBUST, IEEE, PLAIN)
        for start_n in range(diag, tl.minimum(diag + BLOCK_M, N), BLOCK_N):
            acc, l1, l2, m_i = _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base,
                                         v_base, kn_base, t_base, skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
                                         D2, DV, BLOCK_N, CAUSAL, True, ROBUST, IEEE, PLAIN)
    else:
        for start_n in range(0, N, BLOCK_N):
            acc, l1, l2, m_i = _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base,
                                         v_base, kn_base, t_base, skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
                                         D2, DV, BLOCK_N, CAUSAL, True, ROBUST, IEEE, PLAIN)

    acc = acc / l1[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_v[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])
    tl.store(LSE1 + bh * N + offs_m, (m_i + tl.log2(l1)) / 1.4426950408889634, mask=row_ok)
    tl.store(LSE2 + bh * N + offs_m, (m_i + tl.log2(l2)) / 1.4426950408889634, mask=row_ok)


# (BLOCK_M, BLOCK_N, num_warps, num_stages); BLOCK_M a multiple of BLOCK_N (the causal split)
_CONFIGS = [(64, 64, 4, 2), (64, 32, 4, 2), (64, 64, 4, 3), (64, 32, 4, 1), (128, 64, 4, 2), (128, 64, 8, 2),
            (128, 32, 4, 2), (128, 64, 8, 3), (128, 128, 8, 2), (64, 64, 8, 2)]
_TUNED = {}


def _launch(cfg, args, N, BH, meta):
    bm, bn, nw, ns = cfg
    _dka_fwd_kernel[(triton.cdiv(N, bm), BH)](*args, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns, **meta)


def _tune(key, args, N, BH, meta):
    """The fastest of _CONFIGS for this shape (median of 5 timed launches after one warm-up)."""
    best, best_ms = None, float("inf")
    for cfg in _CONFIGS:
        try:
            _launch(cfg, args, N, BH, meta)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record(); _launch(cfg, args, N, BH, meta); en.record(); torch.cuda.synchronize()
                ts.append(st.elapsed_time(en))
            ms = sorted(ts)[2]
        except OutOfResources:
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    if best is None:
        raise RuntimeError(f"no kernel configuration fits for {key}")
    _TUNED[key] = best
    return best


def fused_forward(q, k, v, t, q_sq, k_sq, consts, robust, causal, config=None, plain=False):
    """
    (o1, lse1, lse2) of the fused forward pass. q, k: [B, H, N, 2D]; v: [B, H, N, Dv] (same dtype); t: [B|1, N];
    q_sq, k_sq: [B, H, N] float32; consts: (mu, v_inf, s_inf, inv_nu, coef), each [H] float32. o1 has v's dtype.
    config: (BLOCK_M, BLOCK_N, num_warps, num_stages), else tuned per shape on first use. plain=True (benchmarking
    only): plain softmax attention with scale consts[4] in the same kernel, to separate DKA's cost from the kernel's.
    """
    B, H, N, D2 = q.shape
    DV = v.shape[-1]
    q, k, v = (x if x.stride(-1) == 1 else x.contiguous() for x in (q, k, v))
    t = t.to(torch.float32).expand(B, N).contiguous()
    q_sq, k_sq = (x.to(torch.float32).contiguous() for x in (q_sq, k_sq))
    consts = [c.to(torch.float32).contiguous() for c in consts]
    o = torch.empty(B, H, N, DV, device=q.device, dtype=v.dtype)
    lse1 = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
    lse2 = torch.empty_like(lse1)
    args = (q, k, v, t, q_sq, k_sq, *consts, o, lse1, lse2,
            q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), o.stride(0), o.stride(1), o.stride(2), t.stride(0), H, N)
    meta = dict(D2=D2, DV=DV, CAUSAL=causal, ROBUST=robust, IEEE=q.dtype == torch.float32, PLAIN=plain)
    if config is None:
        key = (N, D2, DV, q.dtype, causal, robust, B * H, torch.cuda.get_device_name(q.device), plain)
        config = _TUNED.get(key) or _tune(key, args, N, B * H, meta)
    _launch(config, args, N, B * H, meta)
    return o, lse1, lse2
