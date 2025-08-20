import os
import pickle
import psutil
import time
import tempfile
import logging
from toknix.bpe import train_bpe

def test_train_bpe_from_file():
    # Create a temporary text file
    text = "hello world\nhello test\nworld test\n" * 1000
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as f:
        f.write(text)
        data_path = f.name
    vocab_size = 200
    special_tokens = ["<|endoftext|>"]
    num_workers = 2
    start_time = time.time()
    vocab, merges = train_bpe(data_path, vocab_size, special_tokens, num_workers)
    elapsed = time.time() - start_time
    # Serialize vocab and merges
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(vocab, f)
        vocab_file = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(merges, f)
        merges_file = f.name
    # Resource usage
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logger = logging.getLogger("toknix.tests.test_bpe_from_file")
    logger.info(f"Training time: {elapsed/3600:.4f} hours ({elapsed:.2f} seconds)")
    logger.info(f"Peak memory usage: {mem_mb:.2f} MB")
    # Longest token in vocab
    longest_token = max(vocab.values(), key=len)
    logger.info(f"Longest token in vocab (len={len(longest_token)}): {repr(longest_token)}")
    # Clean up
    os.remove(data_path)
    os.remove(vocab_file)
    os.remove(merges_file)
    assert isinstance(vocab, dict)
    assert isinstance(merges, list)
    assert elapsed < 30  # Should be fast for small data
