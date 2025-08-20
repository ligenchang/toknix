# toknix/README.md

# toknix

A general-purpose, open-source tokenizer library (BPE and more).

## Installation

```bash
pip install .
```

## Usage (Python)

```python
from toknix.bpe import train_bpe
vocab, merges = train_bpe('data.txt', 32000)
```

## Usage (CLI)

```bash
python -m toknix.cli --data data.txt --vocab-size 32000 --special-tokens <|endoftext|>
```

## License
MIT

pip install -e .
