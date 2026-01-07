#!/usr/bin/env python3
"""
Tokenize a text file using a pre-trained BPE tokenizer.

Example:
    uv run python tokenize_dataset.py \
        --vocab ../data/ts_vocab.pkl \
        --merges ../data/ts_merges.pkl \
        --input_path ../data/TinyStoriesV2-GPT4-valid.txt \
        --output_path ../data/ts_valid_tokens.npz
        
or for openwebtext:
    uv run python tokenize_dataset.py \
        --vocab ../data/owt_vocab.pkl \
        --merges ../data/owt_merges.pkl \
        --input_path ../data/owt_valid.txt \
        --output_path ../data/owt_valid_tokens.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from tokenization.bpe import BPETokenizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a text file using a pre-trained BPE tokenizer.")
    parser.add_argument("--vocab", type=Path, required=True, help="Path to the vocab pickle file.")
    parser.add_argument("--merges", type=Path, required=True, help="Path to the merges pickle file.")
    parser.add_argument("--input_path", type=Path, required=True, help="Path to the text file to tokenize.")
    parser.add_argument("--output_path", type=Path, required=True, help="Path for the output .npz file.")
    parser.add_argument("--special-token", action="append", dest="special_tokens",
        default=["<|endoftext|>"], help="Special token to handle (repeatable). Always includes <|endoftext|> as default.")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading tokenizer from %s and %s", args.vocab, args.merges)
    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    logger.info("Tokenizing %s", args.input_path)
    with open(args.input_path, "r", encoding="utf-8") as f:
        text = f.read()

    encoding = tokenizer.encode(text)
    tokens = np.array(encoding, dtype=np.uint16)

    logger.info("Saving %d tokens to %s", len(tokens), args.output_path)
    np.savez(args.output_path, tokens=tokens)

    print(f"Tokenized {len(tokens)} tokens")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
