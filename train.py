"""
Legacy training script. Use the toknix package or CLI for BPE training.
Example usage:
    python -m toknix.cli --data data/owt_train.txt --vocab-size 32000 --special-tokens <|endoftext|> --num-workers 12
"""

from toknix.cli import main

if __name__ == "__main__":
    main()
