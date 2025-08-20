"""
toknix.chunked_encode
---------------------
Efficient, parallel, chunked file tokenization for large-scale NLP pipelines.

This module provides utilities to split large files into chunks, tokenize them in parallel,
and save the results efficiently. Designed for production and open source use.
"""

import os
import mmap
import pickle
import multiprocessing
import tempfile
import logging
from typing import List, Optional, Tuple
from toknix.tokenizer import Tokenizer

def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
    mini_chunk_size: int = 4096
) -> List[int]:
    """
    Find chunk boundaries in a file, attempting to split at special token boundaries.
    Args:
        file: Opened file object in binary mode.
        desired_num_chunks: Number of chunks to split into.
        split_special_token: Byte string to align chunk boundaries.
        mini_chunk_size: How many bytes to scan at a time for the special token.
    Returns:
        List of byte offsets for chunk boundaries.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(args: Tuple) -> str:
    """
    Tokenize a chunk of a file and write token batches to a temporary file.
    Args:
        args: Tuple of chunk parameters (see encode_file_to_chunks).
    Returns:
        Path to the temporary file containing token batches.
    """
    (
        chunk_idx, start, end, raw_text_path, tmp_dir, vocab_path, merges_path,
        special_token, batch_size, chunk_size
    ) = args
    logger = logging.getLogger("toknix.chunked_encode")
    try:
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[special_token])
        tmp_path = os.path.join(tmp_dir, f"chunk_{chunk_idx}.pkl")
        with open(raw_text_path, "rb") as f, open(tmp_path, "wb", buffering=1024*1024) as out_f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                pos = start
                leftover = b""
                lines_buffer = []
                batch_tokens = []
                while pos < end:
                    read_end = min(pos + chunk_size, end)
                    chunk_bytes = leftover + mm[pos:read_end]
                    # Fast decode: try full, then fallback
                    for i in range(4, -1, -1):
                        try:
                            chunk = chunk_bytes[:len(chunk_bytes)-i].decode("utf-8")
                            leftover = chunk_bytes[len(chunk_bytes)-i:] if i > 0 else b""
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        chunk = chunk_bytes[:-1].decode("utf-8", errors="ignore")
                        leftover = chunk_bytes[-1:]
                    lines_buffer.extend(chunk.splitlines())
                    while len(lines_buffer) >= batch_size:
                        batch = lines_buffer[:batch_size]
                        del lines_buffer[:batch_size]
                        batch_tokens.extend(tokenizer.encode_iterable(batch))
                        # Write in larger batches for fewer pickle calls
                        if len(batch_tokens) >= 10_000:
                            pickle.dump(batch_tokens, out_f, protocol=pickle.HIGHEST_PROTOCOL)
                            batch_tokens.clear()
                    pos = read_end
                if lines_buffer:
                    batch_tokens.extend(tokenizer.encode_iterable(lines_buffer))
                if leftover:
                    try:
                        chunk = leftover.decode("utf-8")
                        batch_tokens.extend(tokenizer.encode_iterable(chunk.splitlines()))
                    except Exception:
                        pass
                if batch_tokens:
                    pickle.dump(batch_tokens, out_f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Chunk {chunk_idx} tokenized: {tmp_path}")
        return tmp_path
    except Exception as e:
        logger.error(f"Error in chunk {chunk_idx}: {e}")
        raise

def encode_file_to_chunks(
    raw_text_path: str,
    vocab_path: str,
    merges_path: str,
    special_token: str,
    output_tokens_path: str,
    num_processes: int = 4,
    batch_size: int = 5000,
    chunk_size: int = 64 * 1024 * 1024,
    tmp_dir: Optional[str] = None,
) -> int:
    """
    Tokenize a large file in parallel using chunking and save the output tokens to disk.
    Args:
        raw_text_path: Path to the input text file (utf-8, one line per sample).
        vocab_path: Path to the pickled vocab file.
        merges_path: Path to the pickled merges file.
        special_token: Special token string to align chunk boundaries.
        output_tokens_path: Path to write the output pickled token batches.
        num_processes: Number of parallel processes to use.
        batch_size: Number of lines to batch per pickle dump.
        chunk_size: Number of bytes to read per chunk.
        tmp_dir: Optional temp directory for intermediate files.
    Returns:
        Total number of tokens written.
    """
    logger = logging.getLogger("toknix.chunked_encode")
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    with open(raw_text_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token.encode("utf-8"))
    chunk_offsets = [(i, start, end) for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
    args = [
        (i, start, end, raw_text_path, tmp_dir, vocab_path, merges_path, special_token, batch_size, chunk_size)
        for i, start, end in chunk_offsets
    ]
    logger.info(f"Tokenizing {len(args)} chunks in parallel with {num_processes} processes...")
    with multiprocessing.Pool(num_processes) as pool:
        tmp_paths = pool.map(pretokenize_chunk, args)
    total_tokens = 0
    with open(output_tokens_path, "wb") as out_f:
        for tmp_path in sorted(tmp_paths, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])):
            with open(tmp_path, "rb") as f:
                try:
                    while True:
                        token_batch = pickle.load(f)
                        pickle.dump(token_batch, out_f)
                        total_tokens += len(token_batch)
                except EOFError:
                    pass
    # Clean up temp files
    for tmp_path in tmp_paths:
        try:
            os.remove(tmp_path)
        except Exception as e:
            logger.warning(f"Could not remove temp file {tmp_path}: {e}")
    logger.info(f"Tokenization complete. {total_tokens} tokens written to {output_tokens_path}.")
    return total_tokens
