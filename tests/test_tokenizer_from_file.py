import os
import pickle
import tempfile
from toknix.chunked_encode import encode_file_to_chunks

def test_tokenizer_from_file():
    # Prepare a temp text file and vocab/merges
    text = "hello world\nhello test\nworld test\n" * 1000
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as f:
        f.write(text)
        raw_text_path = f.name
    # Fake vocab/merges
    from toknix.bpe import train_bpe
    vocab_size = 200
    special_token = "<|endoftext|>"
    vocab, merges = train_bpe(raw_text_path, vocab_size, [special_token], 2)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(vocab, f)
        vocab_path = f.name
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(merges, f)
        merges_path = f.name
    output_tokens_path = tempfile.mktemp()
    num_processes = 2
    total_tokens = encode_file_to_chunks(
        raw_text_path=raw_text_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_token=special_token,
        output_tokens_path=output_tokens_path,
        num_processes=num_processes,
    )
    import logging
    logger = logging.getLogger("toknix.tests.test_tokenizer_from_file")
    logger.info(f"Saved {total_tokens} tokens to {output_tokens_path}.")
    # Clean up
    os.remove(raw_text_path)
    os.remove(vocab_path)
    os.remove(merges_path)
    os.remove(output_tokens_path)
    assert total_tokens > 0
