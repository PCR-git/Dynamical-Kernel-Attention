"""Positional precision at a distance, after fine-tuning: can a model retrieve the token exactly k positions before a
marker, and how sharply does it localize as k grows?

Each sequence of SEQ_LEN tokens is a stream of random tokens (from a fixed set of N_SYMBOLS GPT-2 tokens), so no
content predicts any position. Q queries are written into it, each a triple
    [MARK] [cue_k] [answer]          answer = the token k positions before MARK
with k from OFFSETS and cue_k a token that names k (one per offset). Only the answers are scored. Because the stream
is random, the answer can be found only by position: the model must attend to exactly lag k (+1 from the cue).

Fine-tuning: every pretrained model (or, with --scratch, a fresh one) is trained on this task until convergence
(k uniform over OFFSETS). Evaluation
(test-set seeds, before and after fine-tuning): per k, the accuracy, and the share of answers that are the token at a
neighbouring position k - delta (|delta| = 1, 2, 3..5), i.e. how wide the model's localization is.

usage: python evals/finetune_offset.py <out.txt> <name=run_dir|preset:NAME> [...] [--scratch] [--steps 30000] [--lr 1e-3]
           [--batch-size 16] [--queries 8] [--trials 128] [--window 1000] [--patience 4]
Training runs until the mean loss of a window of steps has not improved by 2% for `patience` windows (or reaches
1e-3), at most --steps; warmup 200 steps, then constant lr. --scratch trains the architecture from random
initialization (preset:NAME builds a preset that has no checkpoint).
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/finetune_offset.py
import math
import sys
import time
import torch
import eval_eviction as ev
from dka.training.presets import build_language_model, make_args
from dka.utils import seed_everything

SEQ_LEN, N_SYMBOLS = 512, 512
OFFSETS = (10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500)


def vocab(seed=0):
    """Disjoint token sets: stream symbols, the marker and one cue per offset (common GPT-2 tokens)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(20000 - 256, generator=g)[:N_SYMBOLS + 1 + len(OFFSETS)] + 256
    return ids[:N_SYMBOLS], int(ids[N_SYMBOLS]), ids[N_SYMBOLS + 1:]


def make_sequence(symbols, mark, cues, n_queries, g, k_index=None):
    """(input_ids [SEQ_LEN], labels [SEQ_LEN] with -100 except at the answers, the offset index of each answer)."""
    x = symbols[torch.randint(N_SYMBOLS, (SEQ_LEN,), generator=g)]
    labels = torch.full_like(x, -100)
    used = torch.zeros(SEQ_LEN, dtype=torch.bool)          # positions holding query tokens
    ks = torch.full((SEQ_LEN,), -1)
    placed, tries = 0, 0
    while placed < n_queries and tries < 200:
        tries += 1
        ki = int(torch.randint(len(OFFSETS), (1,), generator=g)) if k_index is None else k_index
        k = OFFSETS[ki]
        p = int(torch.randint(k, SEQ_LEN - 2, (1,), generator=g))   # MARK at p, answer source at p - k
        if used[p:p + 3].any() or used[p - k]:
            continue
        x[p], x[p + 1], x[p + 2] = mark, cues[ki], x[p - k]
        labels[p + 2] = x[p + 2]
        ks[p + 2] = ki
        used[p:p + 3] = True
        placed += 1
    return x, labels, ks


@torch.no_grad()
def evaluate(model, symbols, mark, cues, trials, dev, batch=16):
    """Per offset index: (accuracy, share at |delta| = 1, 2, 3..5): the prediction equals the token at p - k - delta
    and not the answer."""
    model.eval()
    out = {}
    for ki in range(len(OFFSETS)):
        g = torch.Generator().manual_seed(10_000 + ki)       # the same sequences for every model
        counts = torch.zeros(4)
        total = 0
        for b0 in range(0, trials, batch):
            seqs = [make_sequence(symbols, mark, cues, 1, g, k_index=ki) for _ in range(min(batch, trials - b0))]
            x = torch.stack([s[0] for s in seqs]).to(dev)
            logits, _ = model(x)
            pred = logits[:, :-1].argmax(-1).cpu()            # logits at t predict token t + 1
            for (xs, ys, _), pr in zip(seqs, pred):
                a = int((ys != -100).nonzero()[0])            # answer position; MARK at a - 2, source at a - 2 - k
                src, guess = a - 2 - OFFSETS[ki], int(pr[a - 1])
                total += 1
                if guess == int(xs[a]):
                    counts[0] += 1
                    continue
                for j, deltas in enumerate(((1,), (2,), (3, 4, 5)), start=1):
                    if any(0 <= src - s * d < SEQ_LEN and guess == int(xs[src - s * d])
                           for d in deltas for s in (1, -1)):
                        counts[j] += 1
                        break
        out[ki] = (counts / total).tolist()
    model.train()
    return out


def main():
    argv = sys.argv[1:]
    opt = {"--steps": "30000", "--lr": "1e-3", "--batch-size": "16", "--queries": "8", "--trials": "128",
           "--window": "1000", "--patience": "4"}
    scratch = "--scratch" in argv
    argv = [a for a in argv if a != "--scratch"]
    for k in list(opt):
        if k in argv:
            j = argv.index(k)
            opt[k] = argv[j + 1]
            del argv[j:j + 2]
    out_path, runs = argv[0], [a.split("=", 1) for a in argv[1:]]
    steps, lr, bs, nq, trials = (int(opt["--steps"]), float(opt["--lr"]), int(opt["--batch-size"]),
                                 int(opt["--queries"]), int(opt["--trials"]))
    window, patience = int(opt["--window"]), int(opt["--patience"])
    dev = torch.device("cuda")
    symbols, mark, cues = vocab()

    hdr = f"{'model':<12}{'stage':<8}" + "".join(f"{k:>6}" for k in OFFSETS)
    lines = [f"Offset retrieval accuracy (%) vs k: {'trained from scratch' if scratch else 'fine-tuned'} until the "
             f"loss stops improving (< 2% over {patience} windows of {window} steps; at most {steps}), {bs} sequences "
             f"of {SEQ_LEN} tokens, {nq} queries each, lr {lr}; {trials} test queries per k; steps used in brackets", hdr]
    near = ["", "after fine-tuning: share (%) of answers that are a neighbour at |delta| = 1 / 2 / 3..5", hdr]
    print("\n".join(lines), flush=True)
    for name, path in runs:
        seed_everything(2025)
        if path.startswith("preset:"):                                # architecture only (with --scratch)
            args = make_args(path[len("preset:"):], device=dev)
            model = build_language_model(args).to(dev)
        else:
            ckpt = torch.load(ev.latest(path), map_location="cpu", weights_only=False)
            args = ckpt["args"]
            args.device = dev
            model = build_language_model(args).to(dev)
            if not scratch:
                model.load_state_dict(ckpt["model_state_dict"])

        res0 = evaluate(model, symbols, mark, cues, trials, dev)
        lines.append(f"{name:<12}{'before':<8}" + "".join(f"{100 * res0[i][0]:>6.1f}" for i in range(len(OFFSETS))))
        print(lines[-1], flush=True)

        opt_ = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        warm = 200
        sched = torch.optim.lr_scheduler.LambdaLR(opt_, lambda s: min(1.0, (s + 1) / warm))   # then constant
        g = torch.Generator().manual_seed(2025)
        model.backbone.compile()                                      # fixed training shapes
        model.train()
        t0, run_loss, best, stale = time.time(), 0.0, float("inf"), 0
        for step in range(steps):
            seqs = [make_sequence(symbols, mark, cues, nq, g) for _ in range(bs)]
            x, y = torch.stack([s[0] for s in seqs]).to(dev), torch.stack([s[1] for s in seqs]).to(dev)
            loss, _ = model.loss(x, y, causal=True)
            opt_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_.step()
            sched.step()
            run_loss += loss.item()
            if (step + 1) % window == 0:                              # converged: no 2% gain in `patience` windows
                mean = run_loss / window
                print(f"  {name} step {step + 1}  loss {mean:.4f}  {time.time() - t0:.0f}s", flush=True)
                run_loss = 0.0
                if mean < 0.98 * best:
                    best, stale = mean, 0
                else:
                    stale += 1
                if stale >= patience or mean < 1e-3:
                    break
        model.backbone._compiled_call_impl = None

        res = evaluate(model, symbols, mark, cues, trials, dev)
        lines.append(f"{name:<12}{'after':<8}" + "".join(f"{100 * res[i][0]:>6.1f}" for i in range(len(OFFSETS)))
                     + f"   [{step + 1}]")
        for j, label in ((1, "|delta| = 1"), (2, "|delta| = 2"), (3, "|delta| = 3..5")):
            near.append(f"{name if j == 1 else '':<12}{'':<8}"
                        + "".join(f"{100 * res[i][j]:>6.1f}" for i in range(len(OFFSETS))) + f"   ({label})")
        print(lines[-1], flush=True)
        del model, opt_
        torch.cuda.empty_cache()
        with open(out_path, "w") as f:                                # rewritten after every model
            f.write("\n".join(lines + near) + "\n")


if __name__ == "__main__":
    main()
