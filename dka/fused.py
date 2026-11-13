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
              MASK: tl.constexpr, ROBUST: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr = False,
              APPROX: tl.constexpr = True, FLAT: tl.constexpr = False):
    """One key block: the pair logits and the update of the running maximum and sums. MASK: causal and column masks
    (diagonal blocks, and every block if not causal)."""
    FACTOR: tl.constexpr = CAUSAL and APPROX and not MASK and not PLAIN       # E as an outer product
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
        mu2 = mu * 1.4426950408889634                                           # decay rate in units of log2
        if not FACTOR:
            lag = t_q[:, None] - t_k[None, :]
            lag = tl.maximum(lag, 0.0) if CAUSAL else tl.abs(lag)
        if not APPROX:                # the stage-2 form (precise division, log, log1p, exp per pair), for comparison
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
            w = (log_e - coef * phi - tl.log(sigma)) * 1.4426950408889634
        elif FACTOR:
            # below the diagonal every key precedes every query, so with t_ref the block's last key both factors of
            # E = e^{-mu (t_i - t_ref)} e^{-mu (t_ref - t_j)} are <= 1 (no overflow, also for irregular positions);
            # log2 E is a row plus a column vector, so no per-pair lag is kept (fewer live registers)
            t_ref = tl.max(t_k, 0)
            la = -mu2 * (t_q - t_ref)
            lb = -mu2 * (t_ref - t_k)
            a = tl.exp2(la)
            bk = tl.exp2(lb)
            e = a[:, None] * bk[None, :]
            log2_e = la[:, None] + lb[None, :]
        else:
            log2_e = -mu2 * lag
            e = tl.exp2(log2_e)
        if FLAT:
            # Sigma_n = eta^2 (passed as s_inf) at every lag: x = R^2 inv_nu / eta^2, a constant factor; the constant
            # -log eta^2 is added to the log-sum-exps at the end
            r2 = tl.maximum(k_sq[None, :] + e * e * q_sq[:, None] - 2.0 * e * s, 0.0)
            if ROBUST:
                w = log2_e - coef * tl.log2(1.0 + r2 * (inv_nu / s_inf))
            else:
                w = log2_e - (coef * 1.4426950408889634 * inv_nu / s_inf) * r2
        elif APPROX:
            e2 = e * e
            sigma = s_inf - v_inf * e2
            r2 = tl.maximum(k_sq[None, :] + e2 * q_sq[:, None] - 2.0 * e * s, 0.0)
            x = libdevice.fast_dividef(r2 * inv_nu, sigma)                      # approximate division
            # w in units of log2: log2 E - coef phi log2(e) - log2 Sigma, with phi log2(e) = log2(1 + x) (Student-t)
            if ROBUST:
                w = log2_e - coef * tl.log2(1.0 + x) - tl.log2(sigma)
            else:
                w = log2_e - (coef * 1.4426950408889634) * x - tl.log2(sigma)
    if MASK:
        keep = col_ok[None, :]
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        w = tl.where(keep, w, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(w, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(w - m_new[:, None])
    l1 = l1 * alpha + tl.sum(p, 1)
    if FACTOR:
        l2 = l2 * alpha + a * tl.sum(p * bk[None, :], 1)      # sum_j p E = a_i sum_j p b_j
    elif not PLAIN:
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
                    CAUSAL: tl.constexpr, ROBUST: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr,
                    APPROX: tl.constexpr, FLAT: tl.constexpr):
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
                                         D2, DV, BLOCK_N, CAUSAL, False, ROBUST, IEEE, PLAIN, APPROX, FLAT)
        for start_n in range(diag, tl.minimum(diag + BLOCK_M, N), BLOCK_N):
            acc, l1, l2, m_i = _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base,
                                         v_base, kn_base, t_base, skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
                                         D2, DV, BLOCK_N, CAUSAL, True, ROBUST, IEEE, PLAIN, APPROX, FLAT)
    else:
        for start_n in range(0, N, BLOCK_N):
            acc, l1, l2, m_i = _fwd_tile(acc, l1, l2, m_i, q, q_sq, t_q, offs_m, start_n, K, V, KSQ, T, k_base,
                                         v_base, kn_base, t_base, skn, svn, N, mu, v_inf, s_inf, inv_nu, coef,
                                         D2, DV, BLOCK_N, CAUSAL, True, ROBUST, IEEE, PLAIN, APPROX, FLAT)

    acc = acc / l1[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * son + offs_v[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])
    shift = -tl.log(s_inf) if FLAT else 0.0                 # the constant -log Sigma_n left out of the logits
    tl.store(LSE1 + bh * N + offs_m, (m_i + tl.log2(l1)) / 1.4426950408889634 + shift, mask=row_ok)
    tl.store(LSE2 + bh * N + offs_m, (m_i + tl.log2(l2)) / 1.4426950408889634 + shift, mask=row_ok)


# (BLOCK_M, BLOCK_N, num_warps, num_stages); BLOCK_M a multiple of BLOCK_N (the causal split)
# DKA keeps 4-5 more tile-sized float32 values live than softmax (each costs BLOCK_M BLOCK_N / (32 num_warps)
# registers per thread), so narrow key blocks with 8 warps are the ones that avoid spilling at width 128
_CONFIGS = [(64, 64, 4, 2), (64, 32, 4, 2), (64, 64, 4, 3), (64, 32, 4, 1), (128, 64, 4, 2), (128, 64, 8, 2),
            (128, 32, 4, 2), (128, 64, 8, 3), (128, 128, 8, 2), (64, 64, 8, 2), (128, 32, 8, 2), (128, 32, 8, 3),
            (64, 32, 8, 2), (128, 16, 4, 2), (128, 16, 8, 2), (64, 16, 4, 2)]
_TUNED = {}


def _launch(cfg, args, N, BH, meta):
    bm, bn, nw, ns = cfg
    _dka_fwd_kernel[(triton.cdiv(N, bm), BH)](*args, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, num_stages=ns, **meta)


def _tune(key, args, N, BH, meta):
    """The fastest of _CONFIGS for this shape (median of 20 timed launches after 3 warm-ups)."""
    best, best_ms = None, float("inf")
    for cfg in _CONFIGS:
        try:
            for _ in range(3):
                _launch(cfg, args, N, BH, meta)
            torch.cuda.synchronize()
            ts = []
            for _ in range(20):
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record(); _launch(cfg, args, N, BH, meta); en.record(); torch.cuda.synchronize()
                ts.append(st.elapsed_time(en))
            ms = sorted(ts)[10]
        except OutOfResources:
            continue
        if ms < best_ms:
            best, best_ms = cfg, ms
    if best is None:
        raise RuntimeError(f"no kernel configuration fits for {key}")
    _TUNED[key] = best
    return best


def fused_forward(q, k, v, t, q_sq, k_sq, consts, robust, causal, config=None, plain=False, approx=True,
                  flat=False):
    """
    (o1, lse1, lse2) of the fused forward pass. q, k: [B, H, N, 2D]; v: [B, H, N, Dv] (same dtype); t: [B|1, N];
    q_sq, k_sq: [B, H, N] float32; consts: (mu, v_inf, s_inf, inv_nu, coef), each [H] float32. o1 has v's dtype.
    config: (BLOCK_M, BLOCK_N, num_warps, num_stages), else tuned per shape on first use. flat=True: Sigma_n = eta^2
    at every lag (flat precision), with eta^2 passed as s_inf (v_inf unused). plain=True (benchmarking
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
    meta = dict(D2=D2, DV=DV, CAUSAL=causal, ROBUST=robust, IEEE=q.dtype == torch.float32, PLAIN=plain,
                APPROX=approx or flat, FLAT=flat)
    if config is None:
        key = (N, D2, DV, q.dtype, causal, robust, B * H, torch.cuda.get_device_name(q.device), plain, approx, flat)
        config = _TUNED.get(key) or _tune(key, args, N, B * H, meta)
    _launch(config, args, N, B * H, meta)
    return o, lse1, lse2



# ---------------------------------------------------------------------------------------------------------------------
# Backward pass, lean variant (flat precision, exponential weight): w = log E - kappa R^2 with kappa = tau / (nu eta^2)
# per head (the constant -log eta^2 is added to the log-sum-exps outside the kernel). From fused_reference, with
# Sigma constant and phi = x:
#   G_w = P (dO . v - D + dlse1) + dlse2 Q,   G_R2 = -kappa G_w (0 where R^2 was clamped),   G_s = -2 E G_R2,
#   G_logE = G_w + dlse2 Q + E G_R2 (2 E |q|^2 - 2 s),   dkappa = -sum G_w R^2,   dmu = -sum G_logE lag.
# _lean_bwd_dkdv: one program per key block, looping over the query blocks (dK, dV, d|k|^2, and the per-head sums as one
# partial per program); _lean_bwd_dq: one program per query block, looping over the key blocks (dQ, d|q|^2). Every sum
# is a reduction within one program: no atomic additions.
# ---------------------------------------------------------------------------------------------------------------------

@triton.jit
def _lean_bwd_core(q, k, v, do, q_sq, k_sq, t_q, t_k, lse1, r12, dlse1, dlse2, dd, offs_m, offs_n, N, mu2, kappa,
                   CAUSAL: tl.constexpr, MASK: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr = False):
    """
    One tile of the lean backward pass: recomputes the pair terms and returns p (the softmax weights), g_w, g_r2,
    g_s, E, R^2, s and the lag. The weights of the second log-sum-exp are Q = P E r with r = e^{lse1 - lse2} per row,
    so G_w = P (dO . v - D + dlse1 + dlse2 r E) needs one exponential per pair. MASK: causal and bounds masks
    (diagonal tiles and partial blocks); elsewhere every pair of the tile is visible.
    """
    if IEEE:
        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
    else:
        s = tl.dot(q, tl.trans(k))
        dp = tl.dot(do, tl.trans(v))
    if PLAIN:                        # benchmarking only: softmax attention (scale kappa) in the same kernels
        w = s * (kappa * 1.4426950408889634)
        lag = s * 0.0                # placeholders (unused in this mode)
        e = lag
        r2 = lag
        kept = s == s
    else:
        lag = t_q[:, None] - t_k[None, :]
        lag = tl.maximum(lag, 0.0) if CAUSAL else tl.abs(lag)
        log2_e = -mu2 * lag
        e = tl.exp2(log2_e)
        r2 = k_sq[None, :] + e * e * q_sq[:, None] - 2.0 * e * s
        kept = r2 > 0.0
        r2 = tl.where(kept, r2, 0.0)
        w = log2_e - (kappa * 1.4426950408889634) * r2
    if MASK:
        keep = (offs_n[None, :] < N) & (offs_m[:, None] < N)
        if CAUSAL:
            keep = keep & (offs_m[:, None] >= offs_n[None, :])
        w = tl.where(keep, w, float("-inf"))
    p = tl.exp2(w - lse1[:, None])
    if PLAIN:
        g_w = p * (dp - dd[:, None])
        g_r2 = g_w
        g_s = kappa * g_w
    else:
        g_w = p * (dp - dd[:, None] + dlse1[:, None] + (dlse2 * r12)[:, None] * e)
        g_r2 = tl.where(kept, -kappa * g_w, 0.0)
        g_s = -2.0 * e * g_r2
    return p, g_w, g_r2, g_s, e, r2, s, lag


@triton.jit
def _lean_bwd_dkdv_block(dk, dv, dk_sq, p_mu, p_kappa, start_m, Q, DO, QSQ, T, LSE1, LSE2, DLSE1, DLSE2, DD,
                         q_base, do_base, n_base, t_base, sqn, sdn, k, v, k_sq, t_k, offs_n, N, mu2, kappa,
                         D2: tl.constexpr, DVD: tl.constexpr, BLOCK_M: tl.constexpr, CAUSAL: tl.constexpr,
                         MASK: tl.constexpr, IEEE: tl.constexpr, NEED_MU: tl.constexpr, PLAIN: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DVD)
    row_ok = offs_m < N
    q = tl.load(Q + q_base + offs_m[:, None] * sqn + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + do_base + offs_m[:, None] * sdn + offs_v[None, :], mask=row_ok[:, None], other=0.0)
    q_sq = tl.load(QSQ + n_base + offs_m, mask=row_ok, other=0.0)
    t_q = tl.load(T + t_base + offs_m, mask=row_ok, other=0.0)
    lse1 = tl.load(LSE1 + n_base + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
    lse2 = tl.load(LSE2 + n_base + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
    dlse1 = tl.load(DLSE1 + n_base + offs_m, mask=row_ok, other=0.0)
    dlse2 = tl.load(DLSE2 + n_base + offs_m, mask=row_ok, other=0.0)
    dd = tl.load(DD + n_base + offs_m, mask=row_ok, other=0.0)
    r12 = tl.exp2(lse1 - lse2)
    p, g_w, g_r2, g_s, e, r2, s, lag = _lean_bwd_core(q, k, v, do, q_sq, k_sq, t_q, t_k, lse1, r12, dlse1, dlse2, dd,
                                                      offs_m, offs_n, N, mu2, kappa, CAUSAL, MASK, IEEE, PLAIN)
    if IEEE:
        dv += tl.dot(tl.trans(p), do, input_precision="ieee")
        dk += tl.dot(tl.trans(g_s), q, input_precision="ieee")
    else:
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dk += tl.dot(tl.trans(g_s.to(q.dtype)), q)
    if not PLAIN:
        dk_sq += tl.sum(g_r2, 0)
        p_kappa += tl.sum(-g_w * r2, 0)
        if NEED_MU:                  # mu is learned only with learn_decay
            g_loge = g_w + (dlse2 * r12)[:, None] * p * e + e * g_r2 * (2.0 * e * q_sq[:, None] - 2.0 * s)
            p_mu += tl.sum(-g_loge * lag, 0)
    return dk, dv, dk_sq, p_mu, p_kappa


@triton.jit
def _lean_bwd_dkdv(Q, K, V, T, QSQ, KSQ, DO, LSE1, LSE2, DLSE1, DLSE2, DD, MU, KAPPA,
                   DK, DV, DKSQ, PMU, PKAPPA,
                   sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdb, sdh, sdn, stb, H, N, NKB,
                   D2: tl.constexpr, DVD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   CAUSAL: tl.constexpr, IEEE: tl.constexpr, NEED_MU: tl.constexpr, PLAIN: tl.constexpr = False):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DVD)
    col_ok = offs_n < N
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svn + offs_v[None, :], mask=col_ok[:, None], other=0.0)
    k_sq = tl.load(KSQ + bh * N + offs_n, mask=col_ok, other=0.0)
    t_k = tl.load(T + b * stb + offs_n, mask=col_ok, other=0.0)
    mu2 = tl.load(MU + h) * 1.4426950408889634
    kappa = tl.load(KAPPA + h)
    q_base, do_base, n_base, t_base = b * sqb + h * sqh, b * sdb + h * sdh, bh * N, b * stb

    dk = tl.zeros([BLOCK_N, D2], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, DVD], dtype=tl.float32)
    dk_sq = tl.zeros([BLOCK_N], dtype=tl.float32)
    p_mu = tl.zeros([BLOCK_N], dtype=tl.float32)
    p_kappa = tl.zeros([BLOCK_N], dtype=tl.float32)

    n0 = pid_n * BLOCK_N
    if CAUSAL and n0 + BLOCK_N <= N:
        # query blocks overlapping the key block need the causal mask; those after it see every key of the block
        lo = (n0 // BLOCK_M) * BLOCK_M
        diag_end = tl.minimum(((n0 + BLOCK_N + BLOCK_M - 1) // BLOCK_M) * BLOCK_M, N)
        for start_m in range(lo, diag_end, BLOCK_M):
            dk, dv, dk_sq, p_mu, p_kappa = _lean_bwd_dkdv_block(
                dk, dv, dk_sq, p_mu, p_kappa, start_m, Q, DO, QSQ, T, LSE1, LSE2, DLSE1, DLSE2, DD, q_base, do_base,
                n_base, t_base, sqn, sdn, k, v, k_sq, t_k, offs_n, N, mu2, kappa, D2, DVD, BLOCK_M, CAUSAL, True,
                IEEE, NEED_MU, PLAIN)
        for start_m in range(diag_end, N, BLOCK_M):
            dk, dv, dk_sq, p_mu, p_kappa = _lean_bwd_dkdv_block(
                dk, dv, dk_sq, p_mu, p_kappa, start_m, Q, DO, QSQ, T, LSE1, LSE2, DLSE1, DLSE2, DD, q_base, do_base,
                n_base, t_base, sqn, sdn, k, v, k_sq, t_k, offs_n, N, mu2, kappa, D2, DVD, BLOCK_M, CAUSAL, False,
                IEEE, NEED_MU, PLAIN)
    else:                            # non-causal, or the last (partial) key block: masks throughout
        lo = (n0 // BLOCK_M) * BLOCK_M if CAUSAL else 0
        for start_m in range(lo, N, BLOCK_M):
            dk, dv, dk_sq, p_mu, p_kappa = _lean_bwd_dkdv_block(
                dk, dv, dk_sq, p_mu, p_kappa, start_m, Q, DO, QSQ, T, LSE1, LSE2, DLSE1, DLSE2, DD, q_base, do_base,
                n_base, t_base, sqn, sdn, k, v, k_sq, t_k, offs_n, N, mu2, kappa, D2, DVD, BLOCK_M, CAUSAL, True,
                IEEE, NEED_MU, PLAIN)

    tl.store(DK + b * skb + h * skh + offs_n[:, None] * skn + offs_d[None, :], dk.to(DK.dtype.element_ty),
             mask=col_ok[:, None])
    tl.store(DV + b * svb + h * svh + offs_n[:, None] * svn + offs_v[None, :], dv.to(DV.dtype.element_ty),
             mask=col_ok[:, None])
    tl.store(DKSQ + bh * N + offs_n, dk_sq, mask=col_ok)
    tl.store(PMU + bh * NKB + pid_n, tl.sum(tl.where(col_ok, p_mu, 0.0), 0))
    tl.store(PKAPPA + bh * NKB + pid_n, tl.sum(tl.where(col_ok, p_kappa, 0.0), 0))


@triton.jit
def _lean_bwd_dq_block(dq, dq_sq, start_n, K, V, KSQ, T, k_base, v_base, n_base, t_base, skn, svn, q, do, q_sq, t_q,
                       lse1, r12, dlse1, dlse2, dd, offs_m, N, mu2, kappa,
                       D2: tl.constexpr, DVD: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
                       MASK: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DVD)
    col_ok = offs_n < N
    k = tl.load(K + k_base + offs_n[:, None] * skn + offs_d[None, :], mask=col_ok[:, None], other=0.0)
    v = tl.load(V + v_base + offs_n[:, None] * svn + offs_v[None, :], mask=col_ok[:, None], other=0.0)
    k_sq = tl.load(KSQ + n_base + offs_n, mask=col_ok, other=0.0)
    t_k = tl.load(T + t_base + offs_n, mask=col_ok, other=0.0)
    p, g_w, g_r2, g_s, e, r2, s, lag = _lean_bwd_core(q, k, v, do, q_sq, k_sq, t_q, t_k, lse1, r12, dlse1, dlse2, dd,
                                                      offs_m, offs_n, N, mu2, kappa, CAUSAL, MASK, IEEE, PLAIN)
    if IEEE:
        dq += tl.dot(g_s, k, input_precision="ieee")
    else:
        dq += tl.dot(g_s.to(k.dtype), k)
    if not PLAIN:
        dq_sq += tl.sum(g_r2 * e * e, 1)
    return dq, dq_sq


@triton.jit
def _lean_bwd_dq(Q, K, V, T, QSQ, KSQ, DO, LSE1, LSE2, DLSE1, DLSE2, DD, MU, KAPPA, DQ, DQSQ,
                 sqb, sqh, sqn, skb, skh, skn, svb, svh, svn, sdb, sdh, sdn, stb, H, N,
                 D2: tl.constexpr, DVD: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 CAUSAL: tl.constexpr, IEEE: tl.constexpr, PLAIN: tl.constexpr = False):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D2)
    offs_v = tl.arange(0, DVD)
    row_ok = offs_m < N
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdn + offs_v[None, :], mask=row_ok[:, None], other=0.0)
    q_sq = tl.load(QSQ + bh * N + offs_m, mask=row_ok, other=0.0)
    t_q = tl.load(T + b * stb + offs_m, mask=row_ok, other=0.0)
    lse1 = tl.load(LSE1 + bh * N + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
    lse2 = tl.load(LSE2 + bh * N + offs_m, mask=row_ok, other=0.0) * 1.4426950408889634
    dlse1 = tl.load(DLSE1 + bh * N + offs_m, mask=row_ok, other=0.0)
    dlse2 = tl.load(DLSE2 + bh * N + offs_m, mask=row_ok, other=0.0)
    dd = tl.load(DD + bh * N + offs_m, mask=row_ok, other=0.0)
    r12 = tl.exp2(lse1 - lse2)
    mu2 = tl.load(MU + h) * 1.4426950408889634
    kappa = tl.load(KAPPA + h)
    k_base, v_base, n_base, t_base = b * skb + h * skh, b * svb + h * svh, bh * N, b * stb

    dq = tl.zeros([BLOCK_M, D2], dtype=tl.float32)
    dq_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    if CAUSAL:
        diag = pid_m * BLOCK_M                   # key blocks before the diagonal: every pair visible (BLOCK_M % BLOCK_N == 0)
        for start_n in range(0, diag, BLOCK_N):
            dq, dq_sq = _lean_bwd_dq_block(dq, dq_sq, start_n, K, V, KSQ, T, k_base, v_base, n_base, t_base, skn, svn,
                                           q, do, q_sq, t_q, lse1, r12, dlse1, dlse2, dd, offs_m, N, mu2, kappa,
                                           D2, DVD, BLOCK_N, CAUSAL, False, IEEE, PLAIN)
        for start_n in range(diag, tl.minimum(diag + BLOCK_M, N), BLOCK_N):
            dq, dq_sq = _lean_bwd_dq_block(dq, dq_sq, start_n, K, V, KSQ, T, k_base, v_base, n_base, t_base, skn, svn,
                                           q, do, q_sq, t_q, lse1, r12, dlse1, dlse2, dd, offs_m, N, mu2, kappa,
                                           D2, DVD, BLOCK_N, CAUSAL, True, IEEE, PLAIN)
    else:
        for start_n in range(0, N, BLOCK_N):
            dq, dq_sq = _lean_bwd_dq_block(dq, dq_sq, start_n, K, V, KSQ, T, k_base, v_base, n_base, t_base, skn, svn,
                                           q, do, q_sq, t_q, lse1, r12, dlse1, dlse2, dd, offs_m, N, mu2, kappa,
                                           D2, DVD, BLOCK_N, CAUSAL, True, IEEE, PLAIN)

    tl.store(DQ + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :], dq.to(DQ.dtype.element_ty),
             mask=row_ok[:, None])
    tl.store(DQSQ + bh * N + offs_m, dq_sq, mask=row_ok)


# Backward configurations (BLOCK_M, BLOCK_N, num_warps, num_stages), tuned per kernel and shape: each program holds
# float32 accumulators of its own block (dK and dV: 2 x BLOCK_N x width; dQ: BLOCK_M x width) besides the tiles it
# streams, so at width 128 only small blocks fit in registers
_BWD_CONFIGS = [(16, 32, 4, 1), (32, 32, 4, 1), (32, 16, 4, 1), (16, 16, 4, 1), (32, 32, 8, 1), (64, 32, 8, 1),
                (32, 64, 8, 1), (16, 64, 4, 1), (64, 16, 4, 1), (64, 64, 4, 2), (32, 32, 4, 2), (64, 32, 4, 2),
                (32, 64, 4, 2)]
_TUNED_BWD = {}


def _time_launch(fn, reps=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); fn(); en.record(); torch.cuda.synchronize()
        ts.append(st.elapsed_time(en))
    return sorted(ts)[len(ts) // 2]


def lean_backward(q, k, v, t, mu, kappa, o, lse1, lse2, do, dlse1, dlse2, causal, config=None, need_mu=True,
                  plain=False):
    """
    Gradients (dq, dk, dv, dmu, dkappa) of the lean fused update (w = log E - kappa R^2): q, k: [B, H, N, 2D]; v, o, do:
    [B, H, N, Dv]; lse1, lse2 (the kernel's, without -log eta^2), dlse1, dlse2: [B, H, N]; mu, kappa: [H].
    config: ((BLOCK_M, BLOCK_N, num_warps, num_stages) of the dK/dV kernel, the same of the dQ kernel); else each is
    tuned per shape on first use.
    """
    B, H, N, D2 = q.shape
    DV = v.shape[-1]
    ieee = q.dtype == torch.float32
    q, k, v, do = (x.contiguous() for x in (q, k, v, do))
    t = t.to(torch.float32).expand(B, N).contiguous()
    q_sq, k_sq = q.float().pow(2).sum(-1).contiguous(), k.float().pow(2).sum(-1).contiguous()
    dd = (do.float() * o.float()).sum(-1).contiguous()
    lse1, lse2, dlse1, dlse2 = (x.float().contiguous() for x in (lse1, lse2, dlse1, dlse2))
    mu, kappa = mu.float().contiguous(), kappa.float().contiguous()
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dq_sq, dk_sq = torch.empty_like(q_sq), torch.empty_like(k_sq)
    strides = (q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2), do.stride(0), do.stride(1), do.stride(2), t.stride(0), H, N)
    common = (q, k, v, t, q_sq, k_sq, do, lse1, lse2, dlse1, dlse2, dd, mu, kappa)
    meta = dict(D2=D2, DVD=DV, CAUSAL=causal, IEEE=ieee, PLAIN=plain)
    partials = {}
    need_mu = bool(need_mu)

    def run_dkdv(cfg):
        bm, bn, nw, ns = cfg
        nkb = triton.cdiv(N, bn)
        p_mu = torch.empty(B * H, nkb, device=q.device, dtype=torch.float32)
        p_kappa = torch.empty_like(p_mu)
        _lean_bwd_dkdv[(nkb, B * H)](*common, dk, dv, dk_sq, p_mu, p_kappa, *strides, nkb, BLOCK_M=bm, BLOCK_N=bn,
                                     num_warps=nw, num_stages=ns, NEED_MU=need_mu, **meta)
        partials["mu"], partials["kappa"] = p_mu, p_kappa

    def run_dq(cfg):
        bm, bn, nw, ns = cfg
        _lean_bwd_dq[(triton.cdiv(N, bm), B * H)](*common, dq, dq_sq, *strides, BLOCK_M=bm, BLOCK_N=bn, num_warps=nw,
                                                  num_stages=ns, **meta)

    if config is None:
        key = (N, D2, DV, q.dtype, causal, B * H, torch.cuda.get_device_name(q.device), need_mu, plain)
        if key not in _TUNED_BWD:
            best = []
            for run in (run_dkdv, run_dq):
                timed = []
                for cfg in _BWD_CONFIGS:
                    if run is run_dq and cfg[0] % cfg[1]:     # the dQ kernel's causal split needs BLOCK_M % BLOCK_N == 0
                        continue
                    try:
                        timed.append((_time_launch(lambda: run(cfg)), cfg))
                    except OutOfResources:
                        continue
                best.append(min(timed)[1])
            _TUNED_BWD[key] = tuple(best)
        config = _TUNED_BWD[key]
    run_dkdv(config[0])
    run_dq(config[1])
    p_mu, p_kappa = partials["mu"], partials["kappa"]
    dq = (dq.float() + 2 * q.float() * dq_sq[..., None]).to(q.dtype)
    dk = (dk.float() + 2 * k.float() * dk_sq[..., None]).to(k.dtype)
    dmu = p_mu.view(B, H, -1).sum(dim=(0, 2)) if need_mu else torch.zeros_like(mu)
    dkappa = p_kappa.view(B, H, -1).sum(dim=(0, 2))
    return dq, dk, dv, dmu, dkappa


class FusedLean(torch.autograd.Function):
    """(o1, lse1, lse2) of the lean update, w = log E - kappa R^2 (without the constant -log eta^2), with the Triton
    forward and backward. Inputs q, k, v [B, H, N, 2D | Dv] (one dtype), t [B|1, N], mu and kappa [H]."""

    @staticmethod
    def forward(ctx, q, k, v, t, mu, kappa, causal):
        q_sq, k_sq = q.float().pow(2).sum(-1), k.float().pow(2).sum(-1)
        one, zero = torch.ones_like(mu, dtype=torch.float32), torch.zeros_like(mu, dtype=torch.float32)
        o, lse1, lse2 = fused_forward(q, k, v, t, q_sq, k_sq, (mu, zero, one, kappa, one), False, causal, flat=True)
        ctx.save_for_backward(q, k, v, t, mu, kappa, o, lse1, lse2)
        ctx.causal = causal
        return o, lse1, lse2

    @staticmethod
    def backward(ctx, do, dlse1, dlse2):
        q, k, v, t, mu, kappa, o, lse1, lse2 = ctx.saved_tensors
        do = torch.zeros_like(o) if do is None else do
        dlse1 = torch.zeros_like(lse1) if dlse1 is None else dlse1
        dlse2 = torch.zeros_like(lse2) if dlse2 is None else dlse2
        need_mu = ctx.needs_input_grad[4]
        dq, dk, dv, dmu, dkappa = lean_backward(q, k, v, t, mu, kappa, o, lse1, lse2, do.to(o.dtype), dlse1, dlse2,
                                                ctx.causal, need_mu=need_mu)
        return dq, dk, dv, None, dmu.to(mu.dtype) if need_mu else None, dkappa.to(kappa.dtype), None



@torch.compiler.disable      # the Triton launches and the tuner stay outside an enclosing torch.compile
def _lean_call(q, k, v, t, mu, kappa, causal):
    return FusedLean.apply(q, k, v, t, mu, kappa, causal)


def triton_update(p, t, Q, K, V, V_rot, cos, sin, config, tau, causal=True):
    """
    The layer's dV (and step) through the fused kernel, lean variant (flat precision, exponential weight): the logit
    -log eta^2 + log E - tau R^2 / (nu eta^2), i.e. kappa = tau / (nu eta^2) in the kernel and -log eta^2 added to both
    log-sum-exps before the null slot is combined. flex_update's interface; q, k, v enter the kernel in the autocast
    dtype (bf16 under autocast), everything else is float32.
    """
    from .flex import combine
    B, N, H, D, _ = Q.shape
    if 2 * D < 16 or (2 * D) & (2 * D - 1):
        raise ValueError(f"kernel='triton' needs 2 head_dim >= 16 and a power of two, got head_dim={D}")
    low = torch.is_autocast_enabled(Q.device.type)
    dtype = torch.get_autocast_dtype(Q.device.type) if low else Q.dtype
    f32 = torch.promote_types(dtype, torch.float32)
    flat = lambda X: X.flatten(-2).transpose(1, 2).to(dtype).contiguous()          # [B, H, N, 2D]
    q, k, v = flat(Q), flat(K), flat(V_rot)
    eta, nu = p["eta_sq"].to(f32), p["nu"].to(f32)
    kappa = (tau.to(f32) if tau is not None else torch.ones_like(nu)) / (nu * eta)
    o1, lse1, lse2 = _lean_call(q, k, v, t.to(torch.float32), p["mu"].to(f32), kappa, causal)
    shift = -torch.log(eta)                                                          # [H]
    o1 = o1.to(f32).transpose(1, 2).reshape(B, N, H, D, 2)
    lse1, lse2 = lse1.transpose(1, 2) + shift, lse2.transpose(1, 2) + shift          # [B, N, H]
    rot = (cos, sin) if config.rotate_values else (None, None)
    return combine(o1, lse1, lse2, p["log_prior_precision"].to(f32), V.to(f32), *rot)
