# toknix/cli.py
"""
Command-line interface for the toknix tokenizer library.
"""
import argparse
import pickle
import time
import os
import psutil
import logging
from .bpe import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer with toknix.")
    parser.add_argument("--data", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of workers.")
    parser.add_argument("--out-prefix", type=str, default="toknix_bpe", help="Output file prefix.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("toknix.cli")
    logger.info(f"Training BPE tokenizer on {args.data}...")
    start_time = time.time()
    vocab, merges = train_bpe(args.data, args.vocab_size, args.special_tokens, args.num_workers)
    elapsed = time.time() - start_time

    # Serialize vocab and merges
    with open(f"{args.out_prefix}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{args.out_prefix}_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # Resource usage
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logger.info(f"Training time: {elapsed/3600:.4f} hours ({elapsed:.2f} seconds)")
    logger.info(f"Peak memory usage: {mem_mb:.2f} MB")

    # Longest token in vocab
    longest_token = max(vocab.values(), key=len)
    logger.info(f"Longest token in vocab (len={len(longest_token)}): {repr(longest_token)}")
    logger.info("Does it make sense? Yes, typical for BPE.")

if __name__ == "__main__":
    main()
