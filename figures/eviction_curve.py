"""
Perplexity against cache size at L = 4096 (Section "Cache Eviction", Figure fig:eviction), into
images/eviction/eviction_curve.pdf (and .png):

    python figures/eviction_curve.py [--dir C:/dka_runs]

Reads the eviction_*.txt / .log sweeps written by eval_eviction.py (rows "model policy [param] cache ppl ...") and
plots perplexity on positions 2048-4096 against kept keys per head: lines for sliding windows (RoPE, ALiBi, RFA,
DKA), calibrated windows (eps sweep, fitted on validation blocks) for ALiBi and DKA, bound windows for DKA, and
dotted horizontal lines at each model's full-context perplexity. No GPU needed.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt

from common import IMAGES

# sweep file -> which (model, policy) rows to take from it
SOURCES = {
    "eviction_sweep.txt": {("B1", "uniform"), ("B2", "uniform"), ("B2", "full")},
    "eviction_b1.txt": {("DKA", "uniform"), ("DKA", "full"), ("DKA", "bound")},     # bound eps 0.3 ... 0.003
    "eviction_b1_others.txt": {("RFA", "uniform"), ("RFA", "full")},
    "eviction_b1_val_part1.log": {("DKA", "cal"), ("RFA", "cal")},
    "eviction_b1_val2.txt": {("ALiBi", "cal")},
}
NAMES = {"B1": "RoPE", "B2": "ALiBi", "ALiBi": "ALiBi", "RFA": "RFA", "DKA": "DKA"}
COLORS = {"RoPE": "#E69F00", "ALiBi": "#0072B2", "RFA": "#CC79A7", "DKA": "#000000"}
ROW = re.compile(r"^(\w+)\s+(?:(full)|(uniform|cal|bound)\s+[\d.e-]+)\s+([\d.]+)\s+([\d.]+)\s")


def read_rows(directory):
    """{(model, policy): [(cache, ppl), ...]} with model names mapped to the paper's."""
    out = {}
    for fname, wanted in SOURCES.items():
        with open(os.path.join(directory, fname), encoding="utf-8") as f:
            for line in f:
                m = ROW.match(line)
                policy = m and (m[2] or m[3])
                if m and (m[1], policy) in wanted:
                    out.setdefault((NAMES[m[1]], policy), []).append((float(m[4]), float(m[5])))
    return {k: sorted(set(v)) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=os.environ.get("DKA_EVICTION_DIR", "C:/dka_runs"))
    cli = p.parse_args()
    rows = read_rows(cli.dir)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for model in ("RoPE", "ALiBi", "RFA", "DKA"):
        c = COLORS[model]
        x, y = zip(*rows[(model, "uniform")])
        ax.plot(x, y, "-o", color=c, ms=3.5, lw=1.4, label=f"{model}, sliding window")
        if (model, "full") in rows and rows[(model, "full")][0][1] < 26.0:     # RoPE's full context is off the axis
            y_full = rows[(model, "full")][0][1]
            ax.axhline(y_full, color=c, ls=":", lw=1.0)
            ax.text(3200, y_full + 0.05, f"{model}, full context", color=c, fontsize=6.5, ha="right", va="bottom")
    for model, marker in (("ALiBi", "^"), ("DKA", "s")):
        x, y = zip(*rows[(model, "cal")])
        ax.plot(x, y, marker, color=COLORS[model], mfc="white", ms=6, mew=1.3, ls="--", lw=0.8,
                label=f"{model}, calibrated")
    x, y = zip(*rows[("DKA", "bound")])
    ax.plot(x, y, "D", color=COLORS["DKA"], ms=5.5, ls="none", label="DKA, bound")

    ax.set_xscale("log")
    ax.set_xlim(48, 3300)
    ax.set_ylim(21.2, 26.0)
    ax.set_xticks([64, 128, 256, 512, 1024, 2048])
    ax.set_xticklabels(["64", "128", "256", "512", "1024", "2048"])
    ax.set_xlabel("keys kept per head")
    ax.set_ylabel("perplexity (positions 2048-4096)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper right", frameon=False)
    fig.tight_layout()

    out_dir = os.path.join(IMAGES, "eviction")
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"eviction_curve.{ext}"), dpi=200)
    print("wrote", os.path.join(out_dir, "eviction_curve.pdf"))
    off = {f"{m} {pol}": [(round(k), v) for k, v in pts if v > 26.0 or k < 48] for (m, pol), pts in rows.items()}
    print("off-axis:", {m: v for m, v in off.items() if v})


if __name__ == "__main__":
    main()
