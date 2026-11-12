"""
Training-time and memory calculator for DKA / RFA / RoPE / ALiBi language models, from the constants that
benchmarks/bench_gpu.py measures on a GPU (default: benchmarks/bench_rtx5080.json).

Time per token (training, bf16, fused kernels):

    t = 6 N / F  +  layers * heads * (L + 1) / 2 * c_pair

    N       parameters: vocab * width (tied embedding) + layers * 16 width^2 (attention width 2 x width: Q, K, V, O
            8 width^2; FFN 4 x width: 8 width^2) + 2 width^2 (input / output projections)
    F       effective dense throughput of the GPU (TFLOP/s; measured on the full model)
    c_pair  cost of one causal query-key pair in one head, forward + backward (ns; measured per model)

Memory (rough): 16 bytes per parameter (fp32 weights, gradients, Adam moments) plus activations per token, scaled
from the benchmark size by layers x width (the loss is computed in chunks, so the logits stay small).

    python benchmarks/train_time.py --layers 12 --width 768 --heads 12 --context 4096 --tokens 2.4e9
    python benchmarks/train_time.py --layers 24 --width 2048 --heads 16 --context 4096 --tokens 20e9 --gpus 8 \\
        --constants bench_l40.json
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = 50257


def params(layers, width):
    return VOCAB * width + layers * 16 * width ** 2 + 2 * width ** 2


def activation_bytes_per_token(c, model):
    """Activation memory per token at the benchmark size, from the slope of peak memory against micro-batch."""
    rows = sorted((r for r in c["models"][model]["full"] if "peak_gb" in r), key=lambda r: r["batch"])
    if len(rows) < 2:
        return None
    a, b = rows[0], rows[-1]
    return (b["peak_gb"] - a["peak_gb"]) * 2**30 / ((b["batch"] - a["batch"]) * c["context"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--tokens", type=float, default=None, help="training tokens (default: 20 per parameter)")
    p.add_argument("--micro-batch", type=int, default=2, help="sequences per micro-batch (for the memory estimate)")
    p.add_argument("--gpus", type=int, default=1, help="data-parallel GPUs (assumes 90%% scaling efficiency)")
    p.add_argument("--models", nargs="+", default=["dka", "rfa", "rope", "alibi"])
    p.add_argument("--constants", default=os.path.join(HERE, "bench_rtx5080.json"),
                   help="bench_gpu.py output for the GPU")
    cli = p.parse_args()

    with open(cli.constants) as f:
        c = json.load(f)
    N = params(cli.layers, cli.width)
    tokens = cli.tokens or 20 * N
    eff = 1.0 if cli.gpus == 1 else 0.9
    pairs = cli.layers * cli.heads * (cli.context + 1) / 2
    s0 = c["size"]
    print(f"constants: {c['gpu']} ({os.path.basename(cli.constants)}; measured at {s0['layers']} layers x "
          f"{s0['width']}, {s0['heads']} heads, context {c['context']})")
    print(f"model: {cli.layers} layers x {cli.width}, {cli.heads} heads (head width {2 * cli.width // cli.heads}), "
          f"{N / 1e6:.1f}M parameters, context {cli.context}, {tokens / 1e9:.2f}B tokens, {cli.gpus} GPU(s)\n")
    print(f"{'model':6s} {'us/token':>9s} {'attn share':>10s} {'tok/s':>8s} {'B tok/day':>9s} {'hours':>7s} "
          f"{'days':>6s} {'memory/GPU':>11s}")
    for m in cli.models:
        k = c["models"].get(m)
        if not k or k.get("F_tflops") is None or k.get("c_pair_ns") is None:
            print(f"{m:6s} (not in {os.path.basename(cli.constants)})")
            continue
        dense = 6 * N / (k["F_tflops"] * 1e12) * 1e6
        attn = pairs * k["c_pair_ns"] * 1e-3
        t = dense + attn
        rate = 1e6 / t * cli.gpus * eff
        hours = tokens / rate / 3600
        act = activation_bytes_per_token(c, m)
        if act is not None:
            act *= (cli.layers * cli.width) / (s0["layers"] * s0["width"])
            mem = (16 * N + act * cli.micro_batch * cli.context) / 2**30
            mem_s = f"{mem:7.1f} GB"
        else:
            mem_s = "n/a"
        print(f"{m:6s} {t:9.1f} {attn / t:10.0%} {rate / 1e3:7.1f}k {rate * 86400 / 1e9:9.2f} {hours:7.1f} "
              f"{hours / 24:6.1f} {mem_s:>11s}")
    print("\nThe attention cost per pair is measured at the benchmark's head width; it grows slowly with head width "
          "(the score's dot product), so re-run bench_gpu.py at the target size for a final estimate.")


if __name__ == "__main__":
    main()
