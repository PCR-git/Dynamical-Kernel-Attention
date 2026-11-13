# GPU benchmark and training-time calculator

`bench_gpu.py` measures, on one GPU, the two constants that set training time: the attention cost per
query-key pair of each model (DKA, RFA, RoPE, ALiBi) and the effective dense throughput, plus peak memory.
`train_time.py` turns them into hours and memory for any size, context, token budget and number of GPUs.
`bench_rtx5080.json` holds the constants measured on the RTX 5080 (the default for `train_time.py`).

## Running the benchmark on a lab server

1. **Get the code and an environment** (Python >= 3.11, PyTorch >= 2.5 with CUDA; tested with 2.11 / CUDA 12.8):

   ```bash
   git clone https://github.com/PCR-git/Dynamical-Kernel-Attention.git
   cd Dynamical-Kernel-Attention
   conda create -n dka python=3.11 -y && conda activate dka
   pip install torch --index-url https://download.pytorch.org/whl/cu128    # or the CUDA version the driver supports
   pip install -r requirements.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   The benchmark uses random tokens: no dataset or tokenizer download is needed. `torch.compile` needs a C compiler
   (`gcc`) on the path; Linux wheels of PyTorch include Triton.

2. **Pick an idle GPU.** The timings are only meaningful on a GPU nothing else is using:

   ```bash
   nvidia-smi                      # choose a GPU with no processes and ~0 MiB used
   export CUDA_VISIBLE_DEVICES=6   # e.g. one of the A100s on server 1
   ```

3. **Run it** (about 20-30 minutes; each configuration runs in its own process, and the first runs include compile
   time):

   ```bash
   python benchmarks/bench_gpu.py --out benchmarks/bench_a100.json 2>&1 | tee benchmarks/bench_a100.log
   ```

   Repeat on an L40 (`--out benchmarks/bench_l40.json`). The default size is 12 layers x 768 with 12 heads at
   context 4096 (153M parameters); `--layers 9` measures the 124M configuration of the PG-19 presets, and
   `--context 1024` the shorter training length. An out-of-memory configuration is recorded as such and skipped.

4. **Send back the JSON and log files** (or commit them): they hold the per-model constants and the raw timings.

## Estimating a run

```bash
python benchmarks/train_time.py --constants benchmarks/bench_l40.json --layers 9 --width 768 --heads 12 \
    --context 4096 --tokens 3.07e9                  # one pass of PG-19 on one L40
python benchmarks/train_time.py --constants benchmarks/bench_l40.json --layers 24 --width 2048 --heads 16 \
    --context 4096 --tokens 20e9 --gpus 8           # a 1B model on 8 GPUs (90% data-parallel efficiency assumed)
```

The model is t = 6 N / F + layers x heads x (L + 1) / 2 x c_pair per token (N parameters, F the effective dense
throughput, c_pair the cost of one causal query-key pair in one head, forward + backward). The memory column is a
rough estimate (16 bytes per parameter plus activations scaled from the benchmark size); the benchmark's measured
peak memory is the reliable number.
