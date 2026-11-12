"""Paired comparison against a reference model at the training length: per test block of L tokens (default 512,
all 547 blocks), the perplexity ratio of each model to the reference minus one, mean +- standard error over
blocks, overall and by target position (0-64, 64-256, 256-512).

usage: python evals/eval_paired.py <out.txt> <ref_name=run_dir> <name=run_dir> [...] [--length L] [--cpu]
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/eval_paired.py
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

BINS = ((0, 64), (64, 256), (256, 512))


@torch.no_grad()
def block_nll(model, blocks, dev):
    """[n_blocks, L-1] per-token NLL."""
    out = []
    for b in blocks:
        b = b.to(dev)
        logits, _ = model(b[None])
        out.append(F.cross_entropy(logits[0, :-1].float(), b[1:], reduction="none").cpu().numpy())
    return np.stack(out)


def main():
    argv = sys.argv[1:]
    L = int(argv[argv.index("--length") + 1]) if "--length" in argv else 512
    cpu = "--cpu" in argv
    argv = [a for i, a in enumerate(argv) if a not in ("--length", "--cpu") and (i == 0 or argv[i - 1] != "--length")]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    from eval_extrapolation import latest_checkpoint, test_blocks
    from dka.training.lm_data import load_tokenizer
    from dka.training.presets import build_language_model
    dev = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    blocks = test_blocks("wikitext103", load_tokenizer(), L, "datasets", 4)
    nll = {}
    for name, run_dir in runs:
        ck = torch.load(latest_checkpoint(run_dir), map_location="cpu", weights_only=False)
        args = ck["args"]
        args.device = dev
        model = build_language_model(args).to(dev).eval()
        model.load_state_dict(ck["model_state_dict"])
        nll[name] = block_nll(model, blocks, dev)
        print(name, "ppl", round(math.exp(nll[name].mean()), 2), flush=True)
    ref = runs[0][0]
    lines = [f"Paired against {ref}, {len(blocks)} test blocks of {L} tokens", "",
             f"{'model':10s}" + "".join(f"{f'{a}-{b}':>12s}" for a, b in BINS) + f"{'all':>12s}  (perplexity)"]
    for name in nll:
        ppl = [math.exp(nll[name][:, a:b].mean()) for a, b in BINS] + [math.exp(nll[name].mean())]
        lines.append(f"{name:10s}" + "".join(f"{p:12.2f}" for p in ppl))
    lines += ["", f"ppl ratio - 1 (%) vs {ref}, mean +- s.e. over blocks (negative = better than {ref})"]
    for name in nll:
        if name == ref:
            continue
        cells = []
        for a, b in BINS + ((0, L),):
            r = 100 * (np.exp(nll[name][:, a:b].mean(1) - nll[ref][:, a:b].mean(1)) - 1)
            cells.append(f"{r.mean():+6.2f}+-{r.std(ddof=1) / math.sqrt(len(r)):.2f}")
        lines.append(f"{name:10s}" + "".join(f"{c:>14s}" for c in cells))
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
