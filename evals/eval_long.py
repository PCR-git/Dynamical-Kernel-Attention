"""Long-context perplexity within books (PG-19 test): the same tokens at every length.

The test split is a stream of books separated by EOS. Books with at least --span tokens (default the largest
length, 32768) are kept, and the first floor(len / span) spans of span tokens of each are taken; at each length L
every span is cut into span / L windows of L tokens, each window one sequence (batch size 1), and the perplexity
covers every next-token prediction in them. Windows never cross a book boundary, and all lengths are scored on the
same tokens (each window's first token is not predicted, so a length scores (L - 1) / L of them).

Models are rebuilt from the args stored in their checkpoints. By default they run with the fused kernels
(dka_kernel='flex': FlexAttention for DKA and RFA, flash / FlexAttention for the baselines; memory linear in the
length, exact in float32); a model whose configuration the fused path does not support (zero-decay heads, kerple,
post-softmax decay) falls back to the explicit layers, in blocks of --query-chunk queries. Results go to
<run dir>/long_context.txt and are printed as a table.

usage: python evals/eval_long.py <run_dir> [...] [--lengths 4096 8192 16384 32768] [--span S] [--max-spans N]
       [--bf16] [--explicit]
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, "evals")]   # run from the repo root: python evals/eval_long.py
import argparse
import math
import os
import time

import numpy as np
import torch

from eval_extrapolation import latest_checkpoint
from dka.training.lm_data import load_tokenizer, pg19_tokens
from dka.training.presets import build_language_model


def book_spans(tokens, eos, span, max_spans=None):
    """[n_spans, span] int64: the first floor(len / span) spans of every book of the stream (books split at EOS)."""
    ends = np.flatnonzero(tokens == eos)
    starts = np.concatenate([[0], ends[:-1] + 1])
    spans = []
    for s, e in zip(starts, ends):
        for k in range((e - s) // span):
            spans.append(np.asarray(tokens[s + k * span:s + (k + 1) * span], dtype=np.int64))
    spans = torch.from_numpy(np.stack(spans)) if spans else torch.empty(0, span, dtype=torch.long)
    return spans[:max_spans] if max_spans else spans


@torch.no_grad()
def perplexity(model, spans, L, device, bf16):
    total_nll, total = 0.0, 0
    for row in spans:
        for w in row.view(-1, L):
            ids = w[None].to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                loss, _ = model.loss(ids, ids, causal=True)
            total_nll += loss.item() * (L - 1)
            total += L - 1
    return math.exp(total_nll / total)


def load(run_dir, device, fused, query_chunk):
    ckpt_path = latest_checkpoint(run_dir)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.device = device
    kernel = getattr(args, "dka_kernel", "explicit")
    model = None
    if fused:
        args.dka_kernel = "flex"
        try:
            model = build_language_model(args)
        except ValueError as e:                          # a configuration the fused path does not cover
            print(f"  {os.path.basename(run_dir)}: fused kernels not supported ({e}); explicit layers")
    if model is None:
        args.dka_kernel = kernel if not fused else "explicit"
        model = build_language_model(args)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    for m in model.modules():
        if hasattr(m, "query_chunk"):
            m.query_chunk = query_chunk
    return model, ckpt_path, args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories written by train_lm.py")
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--span", type=int, default=None, help="tokens per span (default: the largest length)")
    p.add_argument("--max-spans", type=int, default=None, help="use only the first N spans (quick runs)")
    p.add_argument("--bf16", action="store_true", help="bf16 autocast (default float32)")
    p.add_argument("--explicit", action="store_true", help="the explicit layers instead of the fused kernels")
    p.add_argument("--query-chunk", type=int, default=2048, help="explicit layers: queries per block")
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--gpu", type=int, default=0)
    cli = p.parse_args()
    span = cli.span or max(cli.lengths)
    if any(span % L for L in cli.lengths):
        raise ValueError(f"every length must divide the span ({span})")

    device = torch.device(f"cuda:{cli.gpu}" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    spans = book_spans(pg19_tokens("test", tokenizer, cli.data_dir), tokenizer.eos_token_id, span, cli.max_spans)
    print(f"PG-19 test: {len(spans)} spans of {span} tokens within books ({spans.numel() / 1e6:.1f}M tokens)")

    table = {}
    for run_dir in cli.runs:
        model, ckpt_path, args = load(run_dir, device, not cli.explicit, cli.query_chunk)
        name = os.path.basename(os.path.normpath(run_dir)).split("__")[0]
        table[name] = {}
        lines = [f"# {ckpt_path} (preset {args.preset}); {len(spans)} spans of {span} tokens within PG-19 test books; "
                 f"{'bf16' if cli.bf16 else 'float32'}, {'explicit' if cli.explicit else 'fused'} kernels",
                 f"{'length':>6} {'test_ppl':>9} {'time_s':>7}"]
        for L in cli.lengths:
            t0 = time.time()
            ppl = perplexity(model, spans, L, device, cli.bf16)
            table[name][L] = ppl
            lines.append(f"{L:>6} {ppl:>9.2f} {time.time() - t0:>7.0f}")
            print(f"{name:<16} L={L:<6} ppl {ppl:8.2f}  ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(run_dir, "long_context.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        del model
        torch.cuda.empty_cache()

    print("\n" + f"{'model':<16}" + "".join(f"{'L=' + str(L):>10}" for L in cli.lengths))
    for name, row in table.items():
        print(f"{name:<16}" + "".join(f"{row[L]:>10.2f}" for L in cli.lengths))


if __name__ == "__main__":
    main()
