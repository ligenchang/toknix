"""
Example: Train a BPE tokenizer and use it for encoding/decoding text.
"""
import pickle
from toknix.bpe import train_bpe
from toknix.tokenizer import Tokenizer

if __name__ == "__main__":
    # 1. Train a BPE tokenizer on a text file
    vocab, merges = train_bpe(
        input_path="example_corpus.txt",  # Provide your own file
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
        num_workers=2
    )

    # 2. Save vocab and merges
    with open("example_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("example_merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # 3. Load vocab and merges, create Tokenizer
    with open("example_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("example_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

    # 4. Encode and decode text
    text = "hello world"
    ids = tokenizer.encode(text)
    print("Token IDs:", ids)
    print("Decoded:", tokenizer.decode(ids))
