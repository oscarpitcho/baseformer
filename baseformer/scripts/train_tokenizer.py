#!/usr/bin/env python3
"""
Simple script to train a byte-pair encoder tokenizer.
Special tokens are not added to the vocabulary.

Example:
    uv run python train_tokenizer.py --input_path ../data/TinyStoriesV2-GPT4-train.txt \
                              --output-dir ../data/ \
                              --vocab_size 10048 \
                              --n_processes 16 \
                              --ds_name "ts"
or for openwebtext:
    uv run python train_tokenizer.py --input_path ../data/owt_train.txt \
                            --output-dir ../data/ \
                            --vocab_size 32064 \
                            --n_processes 16 \
                            --ds_name "owt"
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from tokenization.bpe import train_bpe
from tokenization.bpe import BPETokenizer

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a text file.")
    parser.add_argument("--input_path", default="../data/TinyStoriesV2-GPT4-train.txt", type=Path, help="Path to the training text file.")
    parser.add_argument("--ds_name", default="openwebtext", type=str, help="Name of the dataset.")
    parser.add_argument("--vocab_size", default=10000, type=int, help="Final vocabulary size.")
    parser.add_argument("--n_processes", default=1, type=int, help="Number of processes to use for preprocessing.")
    parser.add_argument("--special-token", action="append", dest="special_tokens",
        default=["<|endoftext|>"], help="Special token to consider when doing merging (repeatable). Always includes <|endoftext|> as default.")
    parser.add_argument("--output-dir", type=Path, default=Path("./data/"),
        help="Directory to store vocab and merges pickles. Default: current dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting BPE training on %s with vocab_size=%s", args.input_path, args.vocab_size)
    vocab, merges = train_bpe(
        input_path=str(args.input_path),
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        n_processes=args.n_processes,
    )

    vocab_path = output_dir / f"{args.ds_name}_vocab.pkl"
    merges_path = output_dir / f"{args.ds_name}_merges.pkl"

    # Saving the vocab and merges to files
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    # Save the prettified vocab and merges to files
    prettified_vocab = BPETokenizer.prettify_vocab(vocab)
    prettified_merges = BPETokenizer.prettify_merges(merges)

    with open(output_dir / f"{args.ds_name}_pretty_vocab.json", "w", encoding="utf-8") as f:
        json.dump(prettified_vocab, f, ensure_ascii=False, indent=2)

    with open(output_dir / f"{args.ds_name}_pretty_merges.txt", "w", encoding="utf-8") as f:
        for left, right in prettified_merges:
            f.write(f"{left} {right}\n")

    logger.info("Saved tokenizer artifacts to %s", output_dir)
    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"Vocab size: {len(vocab)}  |  Merges: {len(merges)}")


if __name__ == "__main__":
    main()

