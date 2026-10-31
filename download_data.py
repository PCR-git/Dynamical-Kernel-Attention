"""
Download what train_lm.py needs: WikiText-103 (Hugging Face Salesforce/wikitext, wikitext-103-v1) as parquet
files and the GPT-2 tokenizer.

    python download_data.py                       # -> datasets/wikitext-103/, gpt2_tokenizer/
    python download_data.py --data-dir D --tokenizer-dir T

BabyLM 2025 is not downloaded here (see the README).
"""

import argparse
import os
import shutil
import tempfile

WIKITEXT_FILES = ["train-00000-of-00002.parquet", "train-00001-of-00002.parquet",
                  "validation-00000-of-00001.parquet", "test-00000-of-00001.parquet"]


def download_wikitext103(data_dir):
    from huggingface_hub import hf_hub_download
    out = os.path.join(data_dir, "wikitext-103")
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for name in WIKITEXT_FILES:
            target = os.path.join(out, name)
            if os.path.exists(target):
                print(f"  {target} exists, skipped")
                continue
            path = hf_hub_download("Salesforce/wikitext", f"wikitext-103-v1/{name}", repo_type="dataset",
                                   local_dir=tmp)
            shutil.move(path, target)
            print(f"  {target}")


def save_tokenizer(tokenizer_dir):
    from transformers import GPT2TokenizerFast
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print(f"  {tokenizer_dir} exists, skipped")
        return
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(tokenizer_dir)
    print(f"  {tokenizer_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="datasets")
    p.add_argument("--tokenizer-dir", default="gpt2_tokenizer")
    cli = p.parse_args()
    print("WikiText-103:")
    download_wikitext103(cli.data_dir)
    print("GPT-2 tokenizer:")
    save_tokenizer(cli.tokenizer_dir)


if __name__ == "__main__":
    main()
