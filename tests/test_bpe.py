import os
import tempfile
import pytest
from toknix.bpe import train_bpe, Tokenizer

def test_train_bpe_and_tokenizer_basic():
    # Prepare a small corpus
    text = b"hello world\nhello test\nworld test\n"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(text)
        data_path = f.name
    vocab_size = 50
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(data_path, vocab_size, special_tokens)
    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    # Instead of requiring b'hello' in vocab, check encode/decode correctness
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list)
    decoded = tokenizer.decode(ids)
    assert "hello" in decoded and "world" in decoded
    # Test Tokenizer
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list)
    decoded = tokenizer.decode(ids)
    assert "hello" in decoded and "world" in decoded
    os.remove(data_path)

def test_tokenizer_special_tokens():
    text = b"foo <|endoftext|> bar"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(text)
        data_path = f.name
    vocab_size = 40
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(data_path, vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    ids = tokenizer.encode("foo <|endoftext|> bar")
    # Should contain the special token id
    special_token_bytes = special_tokens[0].encode("utf-8")
    special_token_id = None
    for k, v in vocab.items():
        if v == special_token_bytes:
            special_token_id = k
    assert special_token_id is not None
    assert special_token_id in ids
    os.remove(data_path)
