"""
DKA's learned per-head parameters (Appendix, "Learned Parameters"), into images/dka_params/:

    dka_param_heatmaps.png   final values, layer x head (heads ordered slowest to fastest, labeled by horizon 1/mu_h)
    dka_param_traces.png     the last block's parameters over training, one line per head

    python figures/dka_parameters.py [run_name]          # default D3.b1.r1e4 (DKA, b = 1, base 10^4)

The heatmaps read the trained weights of every block (CPU only). The evidence share S_i / (gamma^-2 + S_i), the
part of each update that comes from the context rather than the null slot, is the training-time batch mean that
train_lm.py logs every 100 steps (history "step"), averaged over the first and the last 10 logs. The traces use the
same history; only the last block's noise parameters are logged during training.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from common import IMAGES, latest_run, load_checkpoint, load_model

LOG_EVERY = 100
N_AVG = 10   # logs averaged for the "initial" and "trained" evidence share


def layer_params(model):
    """Per-layer dicts of [H] numpy arrays: v_inf, eta^2, SNR, nu/d, tau, c_h, horizon."""
    out = []
    for block in model.backbone.blocks:
        attn = block.attn
        with torch.no_grad():
            p = attn.sde_params()
            c = attn.config
            mu, sig, eta = p["mu"], p["sigma_sq"], p["eta_sq"]
            v_inf = sig / (2 * mu)
            out.append({
                "horizon": (1 / mu).numpy(),
                "v_inf": v_inf.numpy(),
                "eta_sq": eta.numpy(),
                "snr": (v_inf / eta).numpy(),
                "nu_over_d": (p["nu"] / c.head_dim).numpy(),
                "tau": torch.exp(attn.log_tau).numpy() if attn.log_tau is not None else np.ones(len(mu)),
                "c_h": torch.exp(attn.log_c).numpy(),
            })
    return out


def heatmaps(params, share_init, share_final, path):
    panels = [  # key or array, title, log color scale
        ("v_inf", r"$v_\infty=\sigma^2/2\mu_h$", True),
        ("eta_sq", r"$\eta^2$", True),
        ("snr", r"$v_\infty/\eta^2$", True),
        ("nu_over_d", r"$\nu/d$", False),
        ("tau", r"$\tau$ (inverse temperature)", False),
        ("c_h", r"$c_h$ (null slot, in agreeing tokens)", True),
        (share_init, "evidence share, initial", False),
        (share_final, "evidence share, trained", False),
    ]
    horizons = params[0]["horizon"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), layout="constrained")
    for ax, (key, title, log) in zip(axes.flat, panels):
        M = np.stack([lp[key] for lp in params]) if isinstance(key, str) else key
        if key is share_init or key is share_final:
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        else:
            im = ax.imshow(M, cmap="viridis", aspect="auto",
                           norm=LogNorm(vmin=M.min(), vmax=M.max()) if log else None)
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7,
                    color="white" if im.norm(v) < 0.5 else "black")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(horizons)), [f"{h:.0f}" if h >= 1 else f"{h:.1g}" for h in horizons], fontsize=8)
        ax.set_yticks(range(M.shape[0]), [str(i + 1) for i in range(M.shape[0])], fontsize=8)
        ax.set_xlabel("head (horizon $1/\\mu_h$, tokens)", fontsize=9)
        ax.set_ylabel("layer", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def traces(history, path):
    h = {k: np.array(history[k], dtype=float) for k in ("mu", "sigma_tilde", "eta", "nu_over_d")}
    c_h = np.array(history["prior_weight"], dtype=float)[:, -1]      # [n_logged, H], last block
    share = np.array(history["step"], dtype=float)[:, -1]
    horizon = 1 / h["mu"][-1]
    panels = [(h["sigma_tilde"], r"$v_\infty$", True), (h["eta"], r"$\eta^2$", True),
              (h["sigma_tilde"] / h["eta"], r"$v_\infty/\eta^2$", True), (h["nu_over_d"], r"$\nu/d$", False),
              (c_h, r"$c_h$", True), (share, "evidence share", False)]
    steps = np.arange(len(share)) * LOG_EVERY
    colors = plt.cm.viridis(np.linspace(0, 0.95, share.shape[1]))
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), layout="constrained")
    for ax, (data, title, log) in zip(axes.flat, panels):
        for k in range(data.shape[1]):
            ax.plot(steps, data[:, k], color=colors[k], lw=1.2, label=f"{horizon[k]:.3g}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("training step", fontsize=9)
        if log:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes.flat[-1].legend(title="horizon", fontsize=8, title_fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "D3.b1.r1e4"
    run = latest_run(name)
    model, _ = load_model(run, torch.device("cpu"))
    history = load_checkpoint(run)["history"]
    step = np.array(history["step"], dtype=float)                    # [n_logged, layers, H]
    out_dir = os.path.join(IMAGES, "dka_params")
    os.makedirs(out_dir, exist_ok=True)
    params = layer_params(model)
    heatmaps(params, step[:N_AVG].mean(0), step[-N_AVG:].mean(0), os.path.join(out_dir, "dka_param_heatmaps.png"))
    traces(history, os.path.join(out_dir, "dka_param_traces.png"))
    print("saved to", out_dir, "from", run)
