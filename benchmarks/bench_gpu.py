"""
GPU benchmark for planning training runs: measures the constants of benchmarks/train_time.py on this GPU.

All runs use bf16 autocast and the fused kernels (the .fx presets: FlexAttention for DKA, RFA and ALiBi, flash
attention for RoPE), at the target size (default 12 layers x 768, 12 heads):

  1. one attention layer of each model, forward + backward, at 512 to 8192 tokens with 8192 tokens per step,
     fitted as  time = a * tokens + c * pairs,  pairs = batch * heads * N (N + 1) / 2  (causal). c is the cost of one
     query-key pair in one head (c_pair); a is the layer's per-token cost (projections).
  2. the full model, compiled training step at the target context, micro-batches of 1, 2 and 4 sequences:
     microseconds per token, peak memory, and the effective dense throughput
        F = 6 N_params / (time per token - layers * heads * (N + 1) / 2 * c_pair).

Each configuration runs in its own process (a CUDA out-of-memory error can abort the process), and the results are
written to a JSON file that train_time.py reads (--constants). Pause any training on the GPU first.

    python benchmarks/bench_gpu.py --out bench_l40.json
    python benchmarks/bench_gpu.py --out bench_a100.json --models dka rope --context 4096
"""
import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = {"dka": "D3.b1.r1e4.fx", "rfa": "M3.b1.r1e4.fx", "rope": "B1.fx", "alibi": "B2.fx"}
LAYER_LENGTHS = (512, 1024, 2048, 4096, 8192)
LAYER_TOKENS = 8192


def size_overrides(spec):
    return dict(d_e=spec["width"], d_k_total=spec["width"], d_v_total=spec["width"], n_heads=spec["heads"],
                num_blocks=spec["layers"])


def child(spec):
    """One measurement (run in a subprocess); prints RESULT <json>."""
    import warnings
    import torch
    warnings.filterwarnings("ignore")
    sys.path.insert(0, REPO)
    from dka import MultiheadIsotropicDKA, MultiheadIsotropicRFA
    from dka.model.baselines import MultiHeadAttentionLayer
    from dka.training.presets import make_args, build_language_model
    dev = torch.device("cuda")
    B, N = spec["batch"], spec["N"]
    torch.manual_seed(0)

    def timed(fn, warmup, reps):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t0)
        return 1000 * sorted(ts)[len(ts) // 2], torch.cuda.max_memory_allocated() / 2**30

    try:
        if spec["kind"] == "layer":
            args = make_args(PRESETS[spec["model"]], device=dev, batch_size=B, seq_len=N,
                             **dict(size_overrides(spec), num_blocks=1))
            model = build_language_model(args).to(dev)
            kinds = (MultiheadIsotropicDKA, MultiheadIsotropicRFA, MultiHeadAttentionLayer)
            layer = next(m for m in model.modules() if isinstance(m, kinds))
            x = torch.randn(B, N, spec["width"], device=dev, requires_grad=True)

            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = layer(x, causal=True)
                out.float().square().mean().backward()
            ms, gb = timed(step, 3, 7)
        else:
            args = make_args(PRESETS[spec["model"]], device=dev, batch_size=B, seq_len=N, **size_overrides(spec))
            model = build_language_model(args).to(dev)
            spec["params"] = sum(p.numel() for p in model.parameters())
            model.backbone.compile()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
            ids = torch.randint(0, args.vocab_size, (B, N), device=dev)

            def step():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = model.loss(ids, ids, causal=True)
                loss.backward(); opt.step(); opt.zero_grad()
            ms, gb = timed(step, 4, 8)
        print("RESULT " + json.dumps(dict(spec, ms=ms, peak_gb=gb)), flush=True)
    except torch.OutOfMemoryError:
        print("RESULT " + json.dumps(dict(spec, error="out of memory")), flush=True)


def run(spec, timeout):
    try:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", json.dumps(spec)],
                           capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired:
        return dict(spec, error="timeout")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
    return dict(spec, error="crashed: " + " | ".join(tail)[:300])


def fit_layer(rows, heads):
    """Least squares time = a * tokens + c * pairs over the layer runs: (a in us per token, c in ns per pair)."""
    import numpy as np
    ok = [r for r in rows if "ms" in r]
    if len(ok) < 2:
        return None, None
    X = np.array([[r["batch"] * r["N"], r["batch"] * heads * r["N"] * (r["N"] + 1) / 2] for r in ok])
    y = np.array([r["ms"] * 1e-3 for r in ok])
    (a, c), *_ = np.linalg.lstsq(X, y, rcond=None)
    return a * 1e6, c * 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="JSON file for the results (train_time.py --constants)")
    p.add_argument("--models", nargs="+", choices=sorted(PRESETS), default=["dka", "rfa", "rope", "alibi"])
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--context", type=int, default=4096, help="context of the full-model runs")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4], help="micro-batches of the full-model runs")
    p.add_argument("--timeout", type=int, default=1200, help="seconds per configuration")
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    cli = p.parse_args()
    if cli.child:
        return child(json.loads(cli.child))
    if not cli.out:
        p.error("--out is required")

    import torch
    size = dict(layers=cli.layers, width=cli.width, heads=cli.heads)
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, size=size, context=cli.context,
               models={})
    print(f"{res['gpu']}, torch {res['torch']}, {cli.layers} layers x {cli.width}, {cli.heads} heads", flush=True)
    for m in cli.models:
        layer_rows = []
        for N in LAYER_LENGTHS:
            r = run(dict(kind="layer", model=m, N=N, batch=max(1, LAYER_TOKENS // N), **size), cli.timeout)
            layer_rows.append(r)
            print(f"  {m:5s} layer  N={N:5d} B={r['batch']:2d}  " +
                  (f"{r['ms']:7.1f} ms  {r['peak_gb']:5.2f} GB" if "ms" in r else r["error"]), flush=True)
        a, c = fit_layer(layer_rows, cli.heads)
        full_rows = []
        for B in cli.batches:
            r = run(dict(kind="model", model=m, N=cli.context, batch=B, **size), cli.timeout)
            if "ms" in r:
                r["us_per_token"] = 1000 * r["ms"] / (B * cli.context)
                if c is not None:
                    attn = cli.layers * cli.heads * (cli.context + 1) / 2 * c * 1e-3          # us per token
                    r["F_tflops"] = 6 * r["params"] / ((r["us_per_token"] - attn) * 1e-6) / 1e12
            full_rows.append(r)
            print(f"  {m:5s} model  N={cli.context} B={B}  " + (
                f"{r['ms']:7.1f} ms  {r['us_per_token']:6.1f} us/token  {r['peak_gb']:5.2f} GB  "
                f"F={r.get('F_tflops', float('nan')):5.1f} TFLOP/s  ({r['params'] / 1e6:.1f}M)" if "ms" in r
                else r["error"]), flush=True)
        best = min((r for r in full_rows if "F_tflops" in r), key=lambda r: r["us_per_token"], default=None)
        res["models"][m] = dict(a_us_per_token=a, c_pair_ns=c, F_tflops=best and best["F_tflops"],
                                layer=layer_rows, full=full_rows)
        print(f"  {m:5s} c_pair {c if c is None else round(c, 4)} ns, a {a if a is None else round(a, 3)} us/token, "
              f"F {best and round(best['F_tflops'], 1)} TFLOP/s", flush=True)
        with open(cli.out, "w") as f:
            json.dump(res, f, indent=1)
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
