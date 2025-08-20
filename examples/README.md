# Toknix Example Usage

This folder demonstrates how to use the toknix library for training, saving, loading, and using a BPE tokenizer, as well as parallel file tokenization.

## 1. Train and Use a Tokenizer

See `train_and_tokenize.py`:

- Train a BPE tokenizer on a text file
- Save and load vocab/merges
- Encode and decode text

```python
python examples/train_and_tokenize.py
```

## 2. Parallel File Tokenization

See `parallel_file_tokenize.py`:

- Tokenize a large file in parallel using chunked processing
- Save output tokens to disk

```python
python examples/parallel_file_tokenize.py
```

## Requirements
- Place your training text as `example_corpus.txt` in this folder or update the script paths.
- The scripts will create `example_vocab.pkl`, `example_merges.pkl`, and `example_corpus.tokens.pkl`.

## CLI Usage
You can also use the CLI:

```sh
python -m toknix.cli --data example_corpus.txt --vocab-size 1000 --special-tokens "<|endoftext|>" --num-workers 2 --out-prefix example
```
