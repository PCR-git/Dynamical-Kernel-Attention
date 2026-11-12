"""
Far memory test: replace one DKA head's weights at lags >= T with their expansion in E = e^(-mu lag) and measure
the loss change on positions 2048-4096 of 4096-token test blocks, against exact attention (GPU, eager).

For a far key, a_ij = g(R^2 / Sigma) / Sigma with R^2 = |K|^2 + E^2 |Q|^2 - 2 E x, x = Re<Q~_i, K~_j>, and
Sigma_n = Sigma_inf - v_inf E^2. To first order in E (Sigma_n -> Sigma_inf):
    order 0:  a0_j = exp(tau logw(|K_j|^2, Sigma_inf)) / Sigma_inf        (key only: a decaying recurrent state)
    order 1:  a0_j + E beta_j x_ij,  beta_j = 2 tau kappa a0_j / (nu Sigma_inf + |K_j|^2)   (linear attention)
with logw(R^2, S) = -kappa log(1 + R^2 / (nu S)) (Student-t). Both give exactly the hybrid's output when the far
weights enter the same sums. Order-1 weights below a floor (1e-30 of a0) are clipped and counted. Baseline:
eviction at the same T (the far keys removed).

    python figures/dka_far_memory.py <out.txt> [--blocks 12] [--budget-min 45]
"""
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

L, Q0 = 4096, 2048
TS = (128, 256, 512, 1024, 2048)
HEADS = [(3, h) for h in range(6)] + [(5, h) for h in range(4)]       # (layer index, head), priority order


def main():
    argv = sys.argv[1:]
    out_path = argv[0]
    n_blocks = int(argv[argv.index("--blocks") + 1]) if "--blocks" in argv else 12
    budget = 60 * float(argv[argv.index("--budget-min") + 1]) if "--budget-min" in argv else 45 * 60
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import REPO, latest_run, load_model
    import dka.dka_attention as da
    from eval_extrapolation import test_blocks
    from dka.training.lm_data import load_tokenizer

    dev = torch.device("cuda")
    model, _ = load_model(latest_run("D3.b1.r1e4"), dev)
    attns = [b.attn for b in model.backbone.blocks]
    for a in attns:
        a.query_chunk = 1024
    blocks = test_blocks("wikitext103", load_tokenizer(os.path.join(REPO, "gpt2_tokenizer")), L,
                         os.path.join(REPO, "datasets"), 4)[:n_blocks].to(dev)

    st = {"layer": None, "cfg": None, "clipped": 0, "total": 0}
    for i, a in enumerate(attns):
        a.register_forward_pre_hook(lambda m, args, kw=None, i=i: st.__setitem__("layer", i))
    orig_res, orig_rob, orig_tmp = da.dka_residual_norm_sq, da.robust_logits, da.temper_content

    def res(Q, K, E, h, c=1.0):
        st["QKE"] = (Q, K, E)
        return orig_res(Q, K, E, h, c)

    def rob(R_sq, Sigma, nu, d, robust_weight=True, *a, **k):
        st["nu"], st["d"] = nu, d
        return orig_rob(R_sq, Sigma, nu, d, robust_weight, *a, **k)

    def tmp(logits, Sigma, tau):
        out = orig_tmp(logits, Sigma, tau)
        cfg = st["cfg"]
        if cfg is None or st["layer"] != cfg[0]:
            return out
        layer, h, T, kind = cfg
        Q, K, E = st["QKE"]
        p = attns[layer].sde_params()
        mu = p["mu"][h]
        lag = -torch.log(E[..., h].clamp_min(1e-38)) / mu                     # [B|1, nq, N]
        far = lag >= T - 0.5
        if kind == "evict":
            out[..., h] = out[..., h].masked_fill(far, float("-inf"))
            return out
        s_inf = p["sigma_sq"][h] / (2 * mu) + p["eta_sq"][h]
        nu, d = st["nu"][h], st["d"]
        kappa = (nu + d) / d
        k_sq = K[..., h, :, :].pow(2).sum((-1, -2))[:, None, :]                # [B, 1, N]
        a0 = torch.exp(-tau[h] * kappa * torch.log1p(k_sq / (nu * s_inf))) / s_inf
        if kind == "order0":
            a = a0.expand_as(out[..., h])
        else:
            x = torch.einsum("bidc,bjdc->bij", Q[..., h, :, :], K[..., h, :, :])
            beta = 2 * tau[h] * kappa * a0 / (nu * s_inf + k_sq)
            a = a0 + E[..., h] * beta * x
            bad = far & (a <= 1e-30 * a0)
            st["clipped"] += int(bad.sum())
            st["total"] += int(far.sum())
            a = torch.maximum(a, 1e-30 * a0)
        out[..., h] = torch.where(far, torch.log(a), out[..., h])
        return out

    da.dka_residual_norm_sq, da.robust_logits, da.temper_content = res, rob, tmp

    @torch.no_grad()
    def loss():
        total = 0.0
        for b in blocks:
            logits, _ = model(b[None])
            total += F.cross_entropy(logits[0, Q0 - 1:L - 1].float(), b[Q0:], reduction="mean").item()
        return total / len(blocks)

    t0 = time.time()
    base = loss()
    f = open(out_path, "w")
    f.write(f"Far memory test, L = {L}, positions {Q0}-{L}, {n_blocks} test blocks; exact loss {base:.4f}\n")
    f.write("dNLL in milli-nats/token (positive = worse than exact); layer is 1-based; keys kept by the head = T "
            f"(of about {(Q0 + L) // 2} on average)\n")
    f.write(f"{'layer':>5} {'head':>4} {'horizon':>8} {'T':>5} {'evict':>9} {'order0':>9} {'order1':>9} {'clipped1':>9}\n")
    f.flush()
    print(f"exact loss {base:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for layer, h in HEADS:
        horizon = 1 / attns[layer].sde_params()["mu"][h].item()
        for T in TS:
            row = {}
            for kind in ("evict", "order0", "order1"):
                st["cfg"], st["clipped"], st["total"] = (layer, h, T, kind), 0, 0
                row[kind] = 1000 * (loss() - base)
                if kind == "order1":
                    row["clip"] = st["clipped"] / max(st["total"], 1)
            st["cfg"] = None
            f.write(f"{layer + 1:>5} {h:>4} {horizon:>8.0f} {T:>5} {row['evict']:>9.2f} {row['order0']:>9.2f} "
                    f"{row['order1']:>9.2f} {row['clip']:>9.4f}\n")
            f.flush()
            print(f"layer {layer + 1} head {h} T {T}: {row}  ({time.time() - t0:.0f}s)", flush=True)
            if time.time() - t0 > budget:
                f.write("(stopped at the time budget)\n")
                return


if __name__ == "__main__":
    main()
