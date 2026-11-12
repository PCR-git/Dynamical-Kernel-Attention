"""Sliding window vs full context, paired, at length L (default 4096) for trained checkpoints.

Each model is run on the same test blocks twice: with full context and with a sliding window of W keys per
head (default 512, the training length). Reported on positions L/2..L:
  - perplexity with full context and with the window;
  - gain from context beyond the window: mean NLL(window) - NLL(full) per block (> 0 = the model uses the
    extra context), in units of 0.01 nats, +- the standard error over blocks;
  - paired difference against a reference (default: the first model with the window): mean NLL - NLL(ref) per
    block, in units of 0.01 nats (> 0 = worse), +- the standard error over blocks.
Pairing on the same blocks removes most of the variance from the text itself, so small differences can be
judged; the standard error does not include variation across training seeds.

usage: python evals/eval_window.py <out.txt> <name=run_dir> [...] [--length L] [--window W] [--blocks N] [--ref NAME] [--cpu]
       e.g. python evals/eval_window.py window.txt B1=runs/wikitext103/B1__<date> D3=runs/wikitext103/D3__<date>
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/eval_window.py
import math
import sys

import torch

from eval_eviction import (CUT, MultiHeadAttentionLayer, MultiheadIsotropicDKA, MultiheadIsotropicRFA, load_tokenizer,
                           run, test_blocks)
from eval_positions import load


def main():
    argv = sys.argv[1:]
    opt = {"--blocks": 32, "--length": 4096, "--window": 512}
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = int(argv[j + 1])
            del argv[j:j + 2]
    ref = None
    if "--ref" in argv:
        j = argv.index("--ref")
        ref = argv[j + 1]
        del argv[j:j + 2]
    cpu = "--cpu" in argv
    argv = [a for a in argv if a != "--cpu"]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    dev = torch.device("cpu" if cpu else "cuda")
    N, W = opt["--length"], opt["--window"]
    blocks = test_blocks("wikitext103", load_tokenizer(), N, "datasets", 4)[:opt["--blocks"]]
    q0 = N // 2 - 1                                                  # NLL index of target position N/2
    per_block = {}
    lines = [f"Sliding window ({W} keys) vs full context at L = {N}, positions {N // 2}-{N}, "
             f"{len(blocks)} WikiText-103 test blocks"]
    for name, run_dir in runs:
        model = load(run_dir, dev)
        for m in model.modules():
            if isinstance(m, (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer)):
                m.register_forward_pre_hook(lambda mod, inp: CUT.update(layer=mod))
        for setting in ("full", "window"):
            CUT.update(policy=None if setting == "full" else "uniform", W=W)
            with torch.no_grad():
                nll, _ = run(model, blocks, dev)
            per_block[name, setting] = nll[:, q0:].mean(1).double()
        CUT.update(policy=None, W=None)
        del model
        if not cpu:
            torch.cuda.empty_cache()
    ref = ref or runs[0][0]

    def se(x):
        return x.std() / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    lines.append(f"{'model':<10}{'ppl full':>10}{'ppl window':>12}{'gain from context':>22}")
    for name, _ in runs:
        f, w = per_block[name, "full"], per_block[name, "window"]
        g = (w - f) * 100
        lines.append(f"{name:<10}{math.exp(f.mean()):>10.2f}{math.exp(w.mean()):>12.2f}{g.mean():>14.2f} +- {se(g):.2f}")
    lines.append(f"\npaired difference vs {ref} with the window (0.01 nats, > 0 = worse)")
    r = per_block[ref, "window"]
    for (name, setting), x in per_block.items():
        d = (x - r) * 100
        lines.append(f"  {name:<10}{setting:<8}{d.mean():+8.2f} +- {se(d):.2f}")
    print("\n".join(lines), flush=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
