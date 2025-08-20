import time
import logging
from toknix.tokenizer import Tokenizer

def test_tokenizer_encode_speed():
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    text = "hello world " * 10000  # Large input
    start = time.time()
    ids = tokenizer.encode(text)
    elapsed = time.time() - start
    logger = logging.getLogger("toknix.tests.test_tokenizer_perf")
    logger.info(f"Encoding 10000x 'hello world' took {elapsed:.4f} seconds.")
    assert elapsed < 2  # Should be fast for simple byte-level encoding

def test_tokenizer_decode_speed():
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    text = "hello world " * 10000
    ids = tokenizer.encode(text)
    start = time.time()
    decoded = tokenizer.decode(ids)
    elapsed = time.time() - start
    logger = logging.getLogger("toknix.tests.test_tokenizer_perf")
    logger.info(f"Decoding 10000x 'hello world' took {elapsed:.4f} seconds.")
    assert decoded.startswith("hello world")
    assert elapsed < 2
