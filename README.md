# Toknix

A fast, production-ready, open source byte-level BPE tokenizer library for large-scale NLP pipelines.

## Features
- High-performance BPE training and tokenization
- Parallel, chunked file tokenization for massive datasets
- tiktoken-style vocab/merges compatibility (export/import)
- Multiprocessing-safe, robust CLI and Python API
- Logging and progress bars for user feedback
- Example scripts for training, tokenizing, and exporting

## Quick Start


### 1. Install Toknix (from source)
You can install Toknix directly from the project folder using [PEP 517/518] and `pyproject.toml`:

```bash
pip install .
```

Or, for an editable/development install:

```bash
pip install -e .
```

This will use the dependencies listed in `pyproject.toml`.

### 2. Train a BPE tokenizer
```python
from toknix.bpe import train_bpe
vocab, merges = train_bpe(
    input_path="corpus.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    num_workers=4
)
```

### 3. Tokenize a file in parallel
```python
from toknix.chunked_encode import encode_file_to_chunks
encode_file_to_chunks(
    raw_text_path="corpus.txt",
    vocab_path="vocab.pkl",
    merges_path="merges.pkl",
    special_token="<|endoftext|>",
    output_tokens_path="corpus.tokens.pkl",
    num_processes=4
)
```

### 4. Decode tokens
```python
from toknix.tokenizer import Tokenizer
import pickle
with open("vocab.pkl", "rb") as vf:
    vocab = pickle.load(vf)
with open("merges.pkl", "rb") as mf:
    merges = pickle.load(mf)
tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
text = tokenizer.decode(token_ids)
```

## CLI Usage
```
python -m toknix.cli --data corpus.txt --vocab-size 10000 --special-tokens "<|endoftext|>" --num-workers 4 --out-prefix my_tokenizer
```

## License
MIT

## Contributing
Pull requests and issues welcome!
