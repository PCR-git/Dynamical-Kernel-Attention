"""
Does DKA's evidence S_i measure how much the context supports a token? Two tests on the trained model (CPU):

1. Evidence against loss: over WikiText-103 test blocks of 512 tokens, each head's log S_i at query i against the
   loss of predicting token i+1, as a rank (Spearman) correlation per head, raw and partial given position and the
   target token's frequency (rank-regressed out of both).
2. Repeated text: trials from eval_retrieval (a 64-token passage repeated D tokens later, against the same
   positions with a different first passage); the change in each head's log S_i on the second copy (tokens 2..64),
   repeat minus control, paired over trials, at D = 64, 128, 256, 384 within 512 tokens. Heads whose horizon is
   much shorter than D cannot see the first copy, so their evidence should not change.

    python figures/dka_evidence.py [run_name]      # default D3.b1.r1e4; caches C:/dka_runs/evidence_<name>.npz

Outputs into images/dka_evidence/ (one PDF per panel):
    dka_evidence_a_repeat_heads.pdf      change in log S_i per layer and head at D = 128 (heads slowest to fastest)
    dka_evidence_b_repeat_distance.pdf   change in log S_i against D for the heads that respond most, and the
                                         largest change among the others; the loss change on the second copy is
                                         printed (the correlations with loss are printed only)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import IMAGES, REPO, latest_run, load_model   # puts the repo on sys.path
from eval_extrapolation import test_blocks
from eval_retrieval import build_trials
from dka.training.lm_data import load_tokenizer

N, P = 512, 64
N_BLOCKS, N_TRIALS = 128, 64
DISTS = (64, 128, 256, 384)
CACHE = "C:/dka_runs/evidence_{}.npz"


@torch.no_grad()
def forward(model, x):
    """Per-token NLL of the next token [N-1] and log S_i [layers, N, H]."""
    logits, infos = model(x[None], causal=True, return_metadata=True)
    nll = F.cross_entropy(logits[0, :-1].float(), x[1:], reduction="none").numpy()
    logS = np.stack([info["log_evidence"][0].float().numpy() for info in infos])
    return nll, logS


def compute(model, tokenizer):
    blocks = test_blocks("wikitext103", tokenizer, N, os.path.join(REPO, "datasets"), 4)
    stream = blocks.reshape(-1)
    freq = np.bincount(stream.numpy(), minlength=tokenizer.vocab_size + 1)
    nlls, logSs, targets = [], [], []
    for b in blocks[:N_BLOCKS]:
        nll, logS = forward(model, b)
        nlls.append(nll)
        logSs.append(logS[:, :-1])
        targets.append(b[1:].numpy())
    trials, dists = build_trials(stream, N_TRIALS, N, P)
    keep = [k for k, d in enumerate(dists) if d in DISTS]
    dlogS, dnll = [], []                                   # [dist, trial, layers, H], [dist, trial]
    for k in keep:
        row_s, row_n = [], []
        for t in range(N_TRIALS):
            (n_r, s_r), (n_c, s_c) = forward(model, trials[k, t, 0]), forward(model, trials[k, t, 1])
            row_s.append((s_r[:, N - P + 1:] - s_c[:, N - P + 1:]).mean(1))
            row_n.append(n_r[N - P:].mean() - n_c[N - P:].mean())
        dlogS.append(row_s)
        dnll.append(row_n)
    return dict(nll=np.stack(nlls), logS=np.stack(logSs), target_freq=freq[np.stack(targets)],
                dlogS=np.array(dlogS), dnll=np.array(dnll), dists=np.array([dists[k] for k in keep]))


def rank(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(len(x))
    return r


def spearman(x, y, controls=()):
    rx, ry = rank(x), rank(y)
    if controls:
        C = np.column_stack([np.ones(len(x))] + [rank(c) for c in controls])
        rx = rx - C @ np.linalg.lstsq(C, rx, rcond=None)[0]
        ry = ry - C @ np.linalg.lstsq(C, ry, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


INK, MUTED = "#2b2b2b", "#6b6b6b"
LINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # reference categorical slots 1-4


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def panel_heads(m, horizon, D, path):
    """m: [layers, H] mean change in log S_i at distance D."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#1c5cab"])
    v = np.abs(m).max()
    fig, ax = plt.subplots(figsize=(5.2, 3.4), layout="constrained")
    im = ax.imshow(m, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
    for (l, h), x in np.ndenumerate(m):
        ax.text(h, l, f"{x:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(x) > 0.6 * v else INK)
    ax.set_xticks(range(len(horizon)), [f"{x:.0f}" for x in horizon], fontsize=8)
    ax.set_yticks(range(m.shape[0]), [str(l + 1) for l in range(m.shape[0])], fontsize=8)
    style(ax, "head (horizon, tokens)", "layer")
    fig.colorbar(im, ax=ax, shrink=0.85).set_label(r"$\Delta\log S_i$ (repeat $-$ control)", color=INK)
    fig.savefig(path)
    plt.close(fig)


def panel_distance(dlogS, dists, horizon, path, n_top=3):
    """dlogS: [dist, trial, layers, H]."""
    m = dlogS.mean(1)
    se = dlogS.std(1) / np.sqrt(dlogS.shape[1])
    flat = m.reshape(len(dists), -1)
    top = np.argsort(-flat.mean(0))[:n_top]
    rest = np.setdiff1d(np.arange(flat.shape[1]), top)
    H = m.shape[-1]
    fig, ax = plt.subplots(figsize=(4.6, 3.4), layout="constrained")
    for c, k in enumerate(top):
        l, h = divmod(k, H)
        ax.errorbar(dists, m[:, l, h], yerr=se[:, l, h], color=LINE_COLORS[c], lw=2, marker="o", ms=5, capsize=2,
                    label=f"layer {l + 1}, horizon {horizon[h]:.0f}")
    ax.plot(dists, np.abs(flat[:, rest]).max(1), color=MUTED, lw=1.5, ls="--", label=f"largest $|\\Delta|$, other {len(rest)} heads")
    ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, "distance between copies $D$ (tokens)", r"$\Delta\log S_i$ (repeat $-$ control)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper right")
    fig.savefig(path)
    plt.close(fig)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "D3.b1.r1e4"
    cache = CACHE.format(name)
    run = latest_run(name)
    model, _ = load_model(run, torch.device("cpu"))
    horizon = 1 / model.backbone.blocks[0].attn.sde_params()["mu"].numpy()
    if os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        d = compute(model, load_tokenizer(os.path.join(REPO, "gpt2_tokenizer")))
        np.savez(cache, **d)
    n_blocks, n_layers, n_pos, H = d["logS"].shape
    nll = d["nll"].reshape(-1)
    pos = np.tile(np.arange(n_pos), n_blocks)
    logf = np.log(d["target_freq"].reshape(-1))
    raw, part = np.zeros((n_layers, H)), np.zeros((n_layers, H))
    for l in range(n_layers):
        for h in range(H):
            s = d["logS"][:, l, :, h].reshape(-1)
            raw[l, h] = spearman(s, nll)
            part[l, h] = spearman(s, nll, (pos, logf))
    np.set_printoptions(precision=2, suppress=True, linewidth=150)
    print("horizons:", np.round(horizon, 3))
    print("Spearman(log S_i, next-token NLL), raw:\n", raw)
    print("partial given position and target frequency:\n", part)
    print("repeat - control, mean over trials: d log S per (distance, layer, head):")
    m, se = d["dlogS"].mean(1), d["dlogS"].std(1) / np.sqrt(d["dlogS"].shape[1])
    for k, D in enumerate(d["dists"]):
        print(f"D = {D}:\n", m[k], "\n s.e.\n", se[k])
    print("repeat - control NLL on the second copy:", np.round(d["dnll"].mean(1), 3), "+-",
          np.round(d["dnll"].std(1) / np.sqrt(d["dnll"].shape[1]), 3))
    out_dir = os.path.join(IMAGES, "dka_evidence")
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    k128 = list(d["dists"]).index(128)
    panel_heads(m[k128], horizon, 128, os.path.join(out_dir, "dka_evidence_a_repeat_heads.pdf"))
    panel_distance(d["dlogS"], d["dists"], horizon, os.path.join(out_dir, "dka_evidence_b_repeat_distance.pdf"))
    print("saved to", out_dir)


if __name__ == "__main__":
    main()
