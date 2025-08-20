import pytest
from toknix.tokenizer import Tokenizer

def test_tokenizer_encode_decode_bytes():
    # Simple vocab and merges for bytes
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    text = "hello world"
    ids = tokenizer.encode(text)
    assert isinstance(ids, list)
    decoded = tokenizer.decode(ids)
    assert decoded == text

def test_tokenizer_with_merges():
    # Merge 'h' and 'e' into a single token
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"he"
    merges = [(b"h", b"e")]
    tokenizer = Tokenizer(vocab, merges)
    text = "hello"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text

def test_tokenizer_special_tokens():
    vocab = {i: bytes([i]) for i in range(256)}
    special_tokens = ["<|endoftext|>"]
    for idx, st in enumerate(special_tokens, start=256):
        vocab[idx] = st.encode("utf-8")
    merges = []
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    text = "foo <|endoftext|> bar"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert "foo" in decoded and "bar" in decoded and "<|endoftext|>" in decoded

def test_tokenizer_empty_string():
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    text = ""
    ids = tokenizer.encode(text)
    assert ids == []
    decoded = tokenizer.decode(ids)
    assert decoded == ""
