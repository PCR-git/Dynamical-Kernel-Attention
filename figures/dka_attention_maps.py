"""
DKA attention maps with the null slot (Appendix, "Attention Maps"), into images/attn_mats/:

    dka_attn_<name>_<start>_to_<end>.png

    python figures/dka_attention_maps.py [run_name] [--length L] [--zoom START]   # default D3.b1.r1e4, L = 4096

Input: the first L-token block of the WikiText-103 test split (CPU). Each panel is one head of the last block,
ordered from slowest to fastest, showing the weight of each value in the update, a_ij E_ij / (gamma^-2 + S_i)
(PowerNorm(gamma=0.2) color scale), and beside it, for each query, the null slot's share gamma^-2 / (gamma^-2 + S_i):
the part of the update that abstains (the evidence share is one minus it). --zoom START shows queries and keys
START..L only (the maps are computed at the full length either way).
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from common import IMAGES, REPO, latest_run, load_model
from eval_extrapolation import test_blocks
from dka.training.lm_data import load_tokenizer


@torch.no_grad()
def last_layer(model, input_ids):
    """Weights [H, N, N], null share [H, N] and horizons 1/mu [H] of the last block."""
    _, infos = model(input_ids, causal=True, return_metadata=True)
    info = infos[-1]
    A = info["attn_mat"][0].permute(2, 0, 1).float().numpy()
    null = info["null_share"][0].T.float().numpy()
    return A, null, 1 / info["mu_v"].float().numpy()


def plot(A, null, horizon, start, end, path):
    fig, axes = plt.subplots(2, 4, figsize=(22, 11.5), layout="constrained")
    q = np.arange(start, end)
    for k, ax in enumerate(axes.flat):
        ax.imshow(A[k, start:end, start:end], cmap="magma", norm=PowerNorm(gamma=0.2, vmin=0, vmax=A[k].max()),
                  extent=(start, end, end, start), interpolation="antialiased", interpolation_stage="data")
        ax.set_title(f"horizon {horizon[k]:.3g} tokens", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        side = make_axes_locatable(ax).append_axes("right", size="12%", pad=0.08)
        side.plot(null[k, start:end], q, color="#0072B2", lw=0.6)
        side.set_ylim(end, start)
        side.set_xlim(0, 1)
        side.set_xticks([0, 1])
        side.set_yticks([])
        side.tick_params(labelsize=10)
        side.set_title("null", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", default="D3.b1.r1e4")
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--zoom", type=int, default=None, help="also plot queries and keys from here to the end")
    p.add_argument("--data-dir", default=os.path.join(REPO, "datasets"))
    cli = p.parse_args()

    tokenizer = load_tokenizer(os.path.join(REPO, "gpt2_tokenizer"))
    L = cli.length
    input_ids = test_blocks("wikitext103", tokenizer, L, cli.data_dir, 4)[0][None]
    model, _ = load_model(latest_run(cli.name), torch.device("cpu"))
    A, null, horizon = last_layer(model, input_ids)
    out_dir = os.path.join(IMAGES, "attn_mats")
    os.makedirs(out_dir, exist_ok=True)
    for start in [0] + ([cli.zoom] if cli.zoom is not None else []):
        path = os.path.join(out_dir, f"dka_attn_{cli.name}_{start}_to_{L}.png")
        plot(A, null, horizon, start, L, path)
        print("saved", path)
    print("mean null share per head:", np.round(null.mean(1), 3))


if __name__ == "__main__":
    main()
