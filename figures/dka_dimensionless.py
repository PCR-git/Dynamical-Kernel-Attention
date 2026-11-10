"""
DKA's learned parameters through dimensionless quantities only (Appendix, "Learned Parameters"), one PDF per panel
into images/dka_params/ (assembled into one figure in LaTeX):

    dka_params_a_snr_traces.pdf          v_inf / eta^2 of the last block over training, one line per head
    dka_params_b_precision_vs_lag.pdf    eta^2 / Sigma_n against lag for the last block's heads: the precision a
                                         key loses to the forcing (the attenuation E^2 comes on top of it)
    dka_params_c_student_t.pdf           log(1 + u) / u, the Student-t log-weight relative to its Gaussian limit,
                                         against u = R^2 / (nu Sigma_n), with the measured range of u shaded
    dka_params_d_logit_decomposition.pdf each head's logit variance split, in order, into lag (recency), key
                                         (salience), query (abstention: the query term moves the row against the
                                         fixed null slot) and the query-key interaction (matching)

    python figures/dka_dimensionless.py [run_name]       # default D3.b1.r1e4

Panels c and d use 8 WikiText-103 test blocks of 512 tokens (CPU, a few minutes); pairs are weighted by their
weight in the update, a_ij E_ij / (gamma^-2 + S_i), so pairs that do not contribute do not count.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import IMAGES, REPO, latest_run, load_checkpoint, load_model   # puts the repo on sys.path
import dka.dka_attention as da
from eval_extrapolation import test_blocks
from dka.training.lm_data import load_tokenizer

LOG_EVERY = 100
N_BLOCKS, N = 8, 512
HEAD_COLORS = ["#0d366b", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#86b6ef"]  # slow->fast
PART_COLORS = {"recency (lag)": "#2a78d6", "salience (key)": "#eb6834", "abstention (query)": "#1baf7a",
               "matching (interaction)": "#eda100"}
INK, MUTED = "#2b2b2b", "#6b6b6b"

_CAP = []
_robust_logits = da.robust_logits


def _capture(R_sq, Sigma, nu, d, robust_weight=True, *args, **kwargs):
    L = _robust_logits(R_sq, Sigma, nu, d, robust_weight, *args, **kwargs)
    _CAP.append({"u": (R_sq / (nu * Sigma)).detach(), "L": L.detach()})
    return L


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax.set_axisbelow(True)


def head_legend(ax, horizon, **kw):
    ax.legend([f"{h:.3g}" for h in horizon], title="horizon (tokens)", frameon=False, fontsize=8, title_fontsize=8,
              labelcolor=INK, **kw)


def panel_a(history, path):
    snr = np.array(history["sigma_tilde"], dtype=float) / np.array(history["eta"], dtype=float)
    horizon = 1 / np.array(history["mu"], dtype=float)[-1]
    steps = np.arange(len(snr)) * LOG_EVERY
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    for k in range(snr.shape[1]):
        ax.plot(steps, snr[:, k], color=HEAD_COLORS[k], lw=2)
    ax.set_yscale("log")
    style(ax, "training step", r"$v_\infty/\eta^2$")
    head_legend(ax, horizon, loc="lower left", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def panel_b(attn, path):
    p = attn.sde_params()
    mu, snr = p["mu"].numpy(), (p["sigma_sq"] / (2 * p["mu"]) / p["eta_sq"]).numpy()
    lag = np.logspace(0, np.log10(4096), 300)
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    for k in range(len(mu)):
        ax.plot(lag, 1 / (1 + snr[k] * (1 - np.exp(-2 * mu[k] * lag))), color=HEAD_COLORS[k], lw=2)
    ax.axvline(512, color=MUTED, lw=1, ls="--")
    ax.text(512 * 1.08, 0.902, "training\nlength", color=MUTED, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0.895, 1.005)
    style(ax, "lag (tokens)", r"$\eta^2/\Sigma_n$")
    head_legend(ax, 1 / mu, loc="lower left", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def panel_c(u, w, path):
    order = np.argsort(u)
    cw = np.cumsum(w[order]) / w.sum()
    q10, q50, q90 = (u[order][np.searchsorted(cw, q)] for q in (0.1, 0.5, 0.9))
    x = np.logspace(-2, np.log10(20), 400)
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    ax.axvspan(q10, q90, color="#cde2fb", lw=0, label="measured $u$ (10th-90th pct.)")
    ax.axvline(q50, color="#2a78d6", lw=1.2, ls="--", label=f"median $u$ = {q50:.2f}")
    ax.plot(x, np.log1p(x) / x, color=INK, lw=2, label=r"$\log(1+u)/u$")
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.text(0.012, 1.01, "Gaussian limit", color=MUTED, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.12)
    style(ax, r"$u = R^2/(\nu\Sigma_n)$", "Student-$t$ log-weight / Gaussian")
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.savefig(path)
    plt.close(fig)
    return q10, q50, q90


def ordered_shares(L, mask, lag, w, iters=30):
    """Weighted variance shares of L on causal pairs, adding lag, then key, then query terms (they sum to one)."""
    W = w * mask
    tot_w = W.sum()
    R0 = L - (W * L).sum() / tot_w
    var = lambda X: (W * X ** 2)[mask].sum() / tot_w
    total = var(R0)
    n_lag = lag.max() + 1
    zeros = np.zeros(N)

    def fit(use_key, use_query):
        a, b, c = zeros.copy(), zeros.copy(), np.zeros(n_lag)
        for _ in range(iters):
            num = np.bincount(lag[mask], weights=((R0 - a[:, None] - b[None, :]) * W)[mask], minlength=n_lag)
            c = num / np.maximum(np.bincount(lag[mask], weights=W[mask], minlength=n_lag), 1e-30)
            if use_key:
                b = ((R0 - a[:, None] - c[lag]) * W).sum(0) / np.maximum(W.sum(0), 1e-30)
            if use_query:
                a = ((R0 - b[None, :] - c[lag]) * W).sum(1) / np.maximum(W.sum(1), 1e-30)
        return var(R0 - a[:, None] - b[None, :] - c[lag]) / total

    r_lag, r_key, r_query = fit(False, False), fit(True, False), fit(True, True)
    return np.array([1 - r_lag, r_lag - r_key, r_key - r_query, r_query])


def panel_d(shares, horizon, path):
    """shares: [layers, heads, 4]."""
    n_layers, n_heads, _ = shares.shape
    gap = 1.5
    x = np.array([l * (n_heads + gap) + h for l in range(n_layers) for h in range(n_heads)])
    flat = shares.reshape(-1, 4)
    fig, ax = plt.subplots(figsize=(9.5, 3.4), layout="constrained")
    bottom = np.zeros(len(x))
    for k, (name, color) in enumerate(PART_COLORS.items()):
        ax.bar(x, flat[:, k], bottom=bottom, width=0.85, color=color, edgecolor="#fcfcfb", linewidth=0.6, label=name)
        bottom += flat[:, k]
    centers = [l * (n_heads + gap) + (n_heads - 1) / 2 for l in range(n_layers)]
    ax.set_xticks(centers, [f"layer {l + 1}" for l in range(n_layers)])
    ax.set_xlim(-1, x[-1] + 1)
    ax.set_ylim(0, 1)
    style(ax, f"heads within each layer, slowest (horizon {horizon[0]:.0f}) to fastest ({horizon[-1]:.0f})",
          "share of logit variance")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
    fig.savefig(path)
    plt.close(fig)


@torch.no_grad()
def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "D3.b1.r1e4"
    run = latest_run(name)
    model, _ = load_model(run, torch.device("cpu"))
    history = load_checkpoint(run)["history"]
    out_dir = os.path.join(IMAGES, "dka_params")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})

    panel_a(history, os.path.join(out_dir, "dka_params_a_snr_traces.pdf"))
    panel_b(model.backbone.blocks[-1].attn, os.path.join(out_dir, "dka_params_b_precision_vs_lag.pdf"))

    da.robust_logits = _capture
    tokenizer = load_tokenizer(os.path.join(REPO, "gpt2_tokenizer"))
    blocks = test_blocks("wikitext103", tokenizer, N, os.path.join(REPO, "datasets"), 4)[:N_BLOCKS]
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    mask, lag = j <= i, np.abs(i - j)
    us, ws, shares = [], [], []
    for blk in blocks:
        _CAP.clear()
        _, infos = model(blk[None], causal=True, return_metadata=True)
        per_layer = []
        for cap, info in zip(_CAP, infos):
            Wt, Lg, u = info["attn_mat"][0].numpy(), cap["L"][0].numpy(), cap["u"][0].numpy()
            us.append(u[mask].ravel())
            ws.append(Wt[mask].ravel())
            per_layer.append([ordered_shares(Lg[..., h], mask, lag, Wt[..., h] + 1e-12) for h in range(Wt.shape[-1])])
        shares.append(per_layer)
    da.robust_logits = _robust_logits
    q = panel_c(np.concatenate(us), np.concatenate(ws), os.path.join(out_dir, "dka_params_c_student_t.pdf"))
    horizon = 1 / model.backbone.blocks[0].attn.sde_params()["mu"].numpy()
    shares = np.mean(shares, axis=0)                                       # [layers, heads, 4]
    panel_d(shares, horizon, os.path.join(out_dir, "dka_params_d_logit_decomposition.pdf"))
    print("saved to", out_dir, "from", run)
    print("u (update-weighted) 10/50/90 pct:", np.round(q, 3))
    print("layer-mean shares (recency, salience, abstention, matching):")
    print(np.round(shares.mean(1), 3))


if __name__ == "__main__":
    main()
