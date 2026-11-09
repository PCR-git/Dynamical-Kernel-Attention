"""Tokenized, fixed-length language-modeling datasets (WikiText-103, BabyLM-2025, PG-19)."""

import glob
import os
from itertools import chain

DATASETS = ("wikitext103", "babylm", "pg19")


def load_tokenizer(path="./gpt2_tokenizer/"):
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained(path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _raw_datasets(name, root):
    from datasets import load_dataset
    if name == "wikitext103":
        d = os.path.join(root, "wikitext-103")
        files = {split: os.path.join(d, f"{prefix}-*.parquet")
                 for split, prefix in [("train", "train"), ("validation", "validation"), ("test", "test")]}
        return load_dataset("parquet", data_files=files)
    if name == "babylm":
        d = os.path.join(root, "BabyLM_2025")
        files = {"train": glob.glob(os.path.join(d, "train_100M", "*.train")),
                 "validation": glob.glob(os.path.join(d, "dev", "*.dev")),
                 "test": glob.glob(os.path.join(d, "test", "*.test"))}
        return load_dataset("text", data_files=files)
    raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}")


def _tokenize(examples, tokenizer, skip_blank):
    texts = examples["text"]
    if skip_blank:
        texts = [t for t in texts if len(t.strip()) > 0]
    return tokenizer(texts)


def _group_texts(examples, block_size):
    """Concatenate a batch of tokenized texts and cut it into blocks of block_size tokens."""
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
    result["labels"] = result["input_ids"].copy()
    return result


def load_lm_datasets(name, tokenizer, block_size=512, root="datasets", num_proc=4):
    """DatasetDict with train/validation/test splits of `block_size`-token blocks (as in the training notebooks)."""
    babylm = name == "babylm"
    raw = _raw_datasets(name, root)
    tokenized = raw.map(_tokenize, batched=True, num_proc=num_proc, remove_columns=["text"],
                        fn_kwargs={"tokenizer": tokenizer, "skip_blank": babylm})
    lm = tokenized.map(_group_texts, batched=True, num_proc=num_proc, fn_kwargs={"block_size": block_size})
    if babylm:  # BabyLM's short files leave some partial blocks
        lm = lm.filter(lambda x: len(x["input_ids"]) == block_size)
    return lm


class _Blocks:
    """Token blocks held in one [n_blocks, block_size] tensor; labels mask pad tokens with -100."""

    def __init__(self, input_ids, pad_token_id):
        self.input_ids = input_ids
        self.labels = input_ids.masked_fill(input_ids == pad_token_id, -100)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_dataloaders(lm_datasets, tokenizer, batch_size):
    """
    Loaders of {"input_ids", "labels"} batches: the same batches, in the same shuffle order, as
    DataCollatorForLanguageModeling(mlm=False) on the Arrow datasets (used by the published runs),
    but read from memory, which saves ~10 ms of CPU work per training step.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    loaders = {}
    for split in ("train", "validation", "test"):
        column = lm_datasets[split].with_format("arrow")["input_ids"].combine_chunks()
        lengths = np.unique(column.value_lengths().to_numpy())
        if len(lengths) != 1:
            raise ValueError(f"{split}: token blocks have different lengths {lengths}")
        ids = column.flatten().to_numpy().astype(np.int64).reshape(len(column), lengths[0])
        blocks = _Blocks(torch.from_numpy(ids), tokenizer.pad_token_id)
        loaders[split] = DataLoader(blocks, batch_size=batch_size, shuffle=(split == "train"))
    return loaders


# --- PG-19: books tokenized once into a uint16 token stream, read in fixed blocks ---
# Files from the emozilla/pg19 parquet mirror under <root>/pg19/data (text column "text"); the token streams go to
# <root>/pg19/tokens/<split>.bin, one EOS after each book. GPT-2 ids fit in uint16 (vocabulary 50257).
PG19_SHARDS = {"train": 23, "validation": 1, "test": 1}


def pg19_tokens(split, tokenizer, root="datasets", batch_books=16):
    """
    uint16 memmap of a split's token stream, written on first use (books in file order, EOS after each).
    Streams batch_books books at a time from the parquet files to the token file, so memory stays at a few GB and
    no Arrow cache is written; the fast tokenizer encodes each batch on all cores.
    """
    import numpy as np
    path = os.path.join(root, "pg19", "tokens", f"{split}.bin")
    if not os.path.exists(path):
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(root, "pg19", "data", f"{split}-*.parquet")))
        if len(files) != PG19_SHARDS[split]:
            raise FileNotFoundError(f"PG-19 {split}: {len(files)} of {PG19_SHARDS[split]} parquet files in "
                                    f"{os.path.join(root, 'pg19', 'data')} (download incomplete?)")
        backend, eos = tokenizer.backend_tokenizer, tokenizer.eos_token_id   # no length limit, no attention masks
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n_books = n = 0
        with open(path + ".tmp", "wb") as out:
            for file in files:
                for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_books, columns=["text"]):
                    for enc in backend.encode_batch(batch.column("text").to_pylist(), add_special_tokens=False):
                        ids = np.append(np.asarray(enc.ids, dtype=np.uint16), np.uint16(eos))
                        out.write(ids.tobytes())
                        n_books, n = n_books + 1, n + len(ids)
        os.replace(path + ".tmp", path)
        print(f"PG-19 {split}: {n_books} books, {n:,} tokens -> {path}")
    return np.memmap(path, dtype=np.uint16, mode="r")


def pg19_blocks(split, tokenizer, block_size, root="datasets"):
    """[n_blocks, block_size] int64 tensor of a split's non-overlapping blocks (the trailing remainder dropped)."""
    import numpy as np
    import torch
    tokens = pg19_tokens(split, tokenizer, root)
    n = len(tokens) // block_size
    return torch.from_numpy(np.asarray(tokens[:n * block_size]).astype(np.int64).reshape(n, block_size))


class _StreamBlocks:
    """Block i of a token stream (memmap, read on access); labels mask EOS (= pad) with -100, as _Blocks."""

    def __init__(self, tokens, block_size, pad_token_id):
        self.tokens, self.block_size, self.pad = tokens, block_size, pad_token_id

    def __len__(self):
        return len(self.tokens) // self.block_size

    def __getitem__(self, i):
        import numpy as np
        import torch
        L = self.block_size
        x = torch.from_numpy(np.asarray(self.tokens[i * L:(i + 1) * L]).astype(np.int64))
        return {"input_ids": x, "labels": x.masked_fill(x == self.pad, -100)}


class PartSampler:
    """
    One pass over the training blocks, split into n_parts "epochs": a fixed random permutation of the blocks, of
    which each iteration yields the next part. The per-epoch checkpoints, validation and LR schedule of train_lm.py
    then work unchanged, and no block is seen twice. Set .part to the epoch to start from when resuming.
    n_used < n_blocks keeps the first n_used blocks of the permutation (a token budget; smaller budgets are subsets).
    """

    def __init__(self, n_blocks, n_parts, seed=2025, n_used=None):
        import numpy as np
        if n_used is not None and not n_parts <= n_used <= n_blocks:
            raise ValueError(f"n_used={n_used} must lie in [n_parts={n_parts}, n_blocks={n_blocks}]")
        self.perm = np.random.default_rng(seed).permutation(n_blocks)[:n_used]
        self.size = len(self.perm) // n_parts
        self.n_parts, self.part = n_parts, 0

    def __len__(self):
        return self.size

    def __iter__(self):
        start = (self.part % self.n_parts) * self.size
        self.part += 1
        return iter(self.perm[start:start + self.size].tolist())


def make_pg19_dataloaders(tokenizer, block_size, batch_size, n_parts, root="datasets", max_tokens=None):
    """
    Train loader over one pass of PG-19 in n_parts parts (PartSampler), or over max_tokens training tokens (a
    fixed random subset of the blocks) if given; validation and test in order.
    """
    from torch.utils.data import DataLoader
    loaders = {}
    for split in ("train", "validation", "test"):
        blocks = _StreamBlocks(pg19_tokens(split, tokenizer, root), block_size, tokenizer.pad_token_id)
        n_used = None if max_tokens is None else min(int(max_tokens) // block_size, len(blocks))
        sampler = PartSampler(len(blocks), n_parts, n_used=n_used) if split == "train" else None
        loaders[split] = DataLoader(blocks, batch_size=batch_size, sampler=sampler)
    return loaders
