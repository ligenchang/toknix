"""
Example: Tokenize a large file in parallel using chunked_encode.
"""
import pickle
from toknix.chunked_encode import encode_file_to_chunks


# This script will train a BPE tokenizer on 'example_corpus.txt' and then tokenize it in parallel.

if __name__ == "__main__":


    # 1. Train BPE tokenizer
    from toknix.bpe import train_bpe
    import logging
    vocab, merges = train_bpe(
        input_path="example_corpus.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        num_workers=2
    )
    # Log vocab and merges info for visualization
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("toknix.examples")
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"First 10 vocab items: {[ (k, v.decode('utf-8', 'replace')) for k, v in list(vocab.items())[:10] ]}")
    logger.info(f"Number of merges: {len(merges)}")
    logger.info(f"First 10 merges: {[ (a.decode('utf-8', 'replace'), b.decode('utf-8', 'replace')) for a, b in merges[:10] ]}")
    with open("example_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("example_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # 2. Tokenize file in parallel
    total_tokens = encode_file_to_chunks(
        raw_text_path="example_corpus.txt",
        vocab_path="example_vocab.pkl",
        merges_path="example_merges.pkl",
        special_token="<|endoftext|>",
        output_tokens_path="example_corpus.tokens.pkl",
        num_processes=2
    )
    print(f"Total tokens written: {total_tokens}")

    # Load vocab/merges for decoding
    from toknix.tokenizer import Tokenizer
    import pickle
    with open("example_vocab.pkl", "rb") as vf:
        vocab = pickle.load(vf)
    with open("example_merges.pkl", "rb") as mf:
        merges = pickle.load(mf)
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # Read and decode a sample from the token file
    decoded_lines = []
    with open("example_corpus.tokens.pkl", "rb") as f:
        try:
            while len(decoded_lines) < 5:
                token_batch = pickle.load(f)
                decoded = tokenizer.decode(token_batch)
                decoded_lines.append(decoded)
        except EOFError:
            pass
    print("Sample decoded text from token file:")
    for i, line in enumerate(decoded_lines):
        print(f"[{i}] {line}")
