# toknix/bpe.py
"""
Byte Pair Encoding (BPE) core logic for the toknix tokenizer library.
Tokenizer implementation adapted from tokenizer.py.
"""


from toknix.tokenizer import Tokenizer
from typing import Dict, List, Tuple

import os
import collections
import regex as re
import pathlib
import mmap
import time
import logging
from datetime import datetime
from typing import Union, Set

GPT2_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

def split_text_by_special_tokens(text: str, special_tokens: List[str]) -> List[str]:
    if not special_tokens:
        return [text]
    escaped_tokens = [re.escape(tok) for tok in special_tokens]
    pattern = "(" + "|".join(escaped_tokens) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p]

def pre_tokenize_text_with_special(text: str, special_tokens: List[str]) -> List[str]:
    parts = split_text_by_special_tokens(text, special_tokens)
    tokens = []
    for part in parts:
        if part in special_tokens:
            tokens.append(part)
        else:
            tokens.extend([m.group() for m in GPT2_PATTERN.finditer(part)])
    return tokens

def find_chunk_boundaries(file: mmap.mmap, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
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

def tokenize_chunk(filename: str, start: int, end: int, special_tokens: List[str], token_to_id: Dict[bytes, int]) -> collections.Counter:
    special_tokens_set = set(special_tokens)
    special_tokens_encoded = {token: token.encode("utf-8") for token in special_tokens_set}
    buffer_size = 64 * 1024 * 1024  # 64MB sub-buffers
    word_freqs = collections.Counter()
    with open(filename, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = mm.size()
        chunk_end = min(end, file_size)
        pos = start
        leftover = ""
        total_bytes = chunk_end - start
        processed_bytes = 0
        chunk_id = start // max(1, (file_size // 1000))
        last_logged_percent = -10
        last_logged_mb = 0
        while pos < chunk_end:
            read_end = min(pos + buffer_size, chunk_end)
            chunk_bytes = mm[pos:read_end]
            chunk = leftover + chunk_bytes.decode("utf-8", errors="ignore")
            if read_end < chunk_end:
                for i in range(1, 5):
                    try:
                        chunk_bytes[-i:].decode("utf-8")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    i = 0
                if i > 0:
                    leftover = chunk[-i:]
                    chunk = chunk[:-i]
                else:
                    leftover = ""
            else:
                leftover = ""
            tokens = pre_tokenize_text_with_special(chunk, special_tokens)
            for token in tokens:
                if token in special_tokens_set:
                    token_bytes = special_tokens_encoded[token]
                    token_id = token_to_id.setdefault(token_bytes, len(token_to_id))
                    word_freqs[(token_id,)] += 1
                else:
                    token_bytes = token.encode("utf-8")
                    token_ids = tuple(token_to_id.setdefault(bytes([b]), len(token_to_id)) for b in token_bytes)
                    word_freqs[token_ids] += 1
            processed_bytes += (read_end - pos)
            percent = 100.0 * processed_bytes / total_bytes if total_bytes > 0 else 100.0
            mb_done = processed_bytes // (100 * 1024 * 1024)
            if percent - last_logged_percent >= 10 or mb_done > last_logged_mb or percent == 100.0:
                logging.getLogger("toknix.bpe").info(f"[tokenize_chunk] Chunk {chunk_id}: {percent:.1f}% ({processed_bytes}/{total_bytes} bytes)")
                last_logged_percent = percent
                last_logged_mb = mb_done
            pos = read_end
        mm.close()
    return word_freqs

def parallel_pretokenize(filename: str, special_tokens: List[str], num_workers: int, token_to_id: Dict[bytes, int]) -> collections.Counter:
    file_size = os.path.getsize(filename)
    effective_workers = min(num_workers, max(1, file_size // (1024 * 1024)))
    split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(filename, "rb") as f:
        boundaries = find_chunk_boundaries(f, effective_workers, split_token)
    args = [
        (filename, start, end, special_tokens, token_to_id)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    word_freqs = collections.Counter()
    import concurrent.futures
    max_workers = effective_workers
    logging.getLogger("toknix.bpe").info(f"[parallel_pretokenize] Tokenizing {len(args)} chunks with {max_workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(tokenize_chunk, *arg) for arg in args]
        completed = 0
        total = len(futures)
        for future in concurrent.futures.as_completed(futures):
            word_freqs.update(future.result())
            completed += 1
            logging.getLogger("toknix.bpe").info(f"[parallel_pretokenize] Completed {completed}/{total} chunks.")
    return word_freqs

class PairCounter:
    def __init__(self, word_freqs: collections.Counter, skipped_pairs: Set[Tuple[int, int]] = None, id_to_token: Dict[int, bytes] = None):
        self.pair_freqs = collections.defaultdict(int)
        self.pair_to_words = collections.defaultdict(set)
        self.skipped_pairs = skipped_pairs or set()
        self.id_to_token = id_to_token
        for word, freq in word_freqs.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
                    self.pair_to_words[pair].add(word)
    def update_pairs(self, old_words: Dict[Tuple[int, ...], int], new_words: Dict[Tuple[int, ...], int]) -> None:
        for word, freq in old_words.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs and pair in self.pair_freqs:
                    self.pair_freqs[pair] -= freq
                    if self.pair_freqs[pair] <= 0:
                        del self.pair_freqs[pair]
                if word in self.pair_to_words.get(pair, set()):
                    self.pair_to_words[pair].discard(word)
                    if not self.pair_to_words[pair]:
                        del self.pair_to_words[pair]
        for word, freq in new_words.items():
            if len(word) <= 1:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in self.skipped_pairs:
                    self.pair_freqs[pair] += freq
                    self.pair_to_words[pair].add(word)
    def get_best_pair(self) -> Tuple[Tuple[int, int], int]:
        if not self.pair_freqs:
            return None, 0
        max_freq = max(self.pair_freqs.values())
        best_pairs = [p for p, freq in self.pair_freqs.items() if freq == max_freq]
        if self.id_to_token is not None:
            best_pair = max(best_pairs, key=lambda p: (self.id_to_token[p[0]], self.id_to_token[p[1]]))
        else:
            best_pair = max(best_pairs)
        return best_pair, max_freq
    def add_skipped_pair(self, pair: Tuple[int, int]) -> None:
        self.skipped_pairs.add(pair)
        if pair in self.pair_freqs:
            del self.pair_freqs[pair]

def apply_merge_fast(word_freqs: Dict[Tuple[int, ...], int], best_pair: Tuple[int, int], merged_token_id: int, affected_words=None):
    bp0, bp1 = best_pair
    merged_count = 0
    changed_words = {}
    new_words = {}
    if affected_words is None:
        affected_words = word_freqs.keys()
    for word in affected_words:
        freq = word_freqs[word]
        new_word = []
        i = 0
        merged = False
        while i < len(word):
            if i < len(word) - 1 and word[i] == bp0 and word[i + 1] == bp1:
                new_word.append(merged_token_id)
                i += 2
                merged = True
                merged_count += 1
            else:
                new_word.append(word[i])
                i += 1
        if merged:
            changed_words[word] = freq
            new_word_tuple = tuple(new_word)
            new_words[new_word_tuple] = new_words.get(new_word_tuple, 0) + freq
        else:
            new_words[word] = freq
    return new_words, merged_count, changed_words, set(affected_words)

def train_bpe(
    input_path: Union[str, pathlib.Path],
    vocab_size: int,
    special_tokens: List[str] = [],
    num_workers: int = 8,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    input_path = str(input_path)
    timings = {}
    token_to_id = {bytes([i]): i for i in range(256)}
    id_to_token = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        b_st = st.encode("utf-8")
        if b_st not in token_to_id:
            token_to_id[b_st] = next_id
            id_to_token[next_id] = b_st
            next_id += 1
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Starting pre-tokenization...")
    t0 = time.time()
    word_freqs = parallel_pretokenize(input_path, special_tokens, num_workers, token_to_id)
    timings['pretokenization'] = time.time() - t0
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Pre-tokenization complete. {sum(word_freqs.values())} words. ({timings['pretokenization']:.2f} sec)")
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Protecting special tokens...")
    t0 = time.time()
    timings['special_token_protection'] = time.time() - t0
    skipped_pairs = set()
    merges: List[Tuple[bytes, bytes]] = []
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Initializing pair counter...")
    t0 = time.time()
    pair_counter = PairCounter(word_freqs, skipped_pairs, id_to_token)
    timings['pair_counter_init'] = time.time() - t0
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Pair counter initialized. ({timings['pair_counter_init']:.2f} sec)")
    merge_iterations = 0
    logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Starting BPE merge loop (target vocab size: {vocab_size})...")
    t0 = time.time()
    update_pairs_total_time = 0.0
    get_best_pair_total_time = 0.0
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        tqdm = None
        use_tqdm = False
    merge_bar = tqdm(total=vocab_size - len(token_to_id), desc="[BPE] Merges", unit="merge") if use_tqdm else None
    while len(token_to_id) < vocab_size:
        t_get_best = time.time()
        best_pair, max_freq = pair_counter.get_best_pair()
        get_best_pair_total_time += time.time() - t_get_best
        if not best_pair or max_freq == 0:
            logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] No more pairs to merge at iteration {merge_iterations}.")
            # If stopped early, fill the bar to 100%
            if merge_bar:
                remaining = (merge_bar.total or 0) - merge_bar.n
                if remaining > 0:
                    merge_bar.update(remaining)
            break
        if merge_iterations % 100 == 0 or len(token_to_id) >= vocab_size:
            logging.getLogger("toknix.bpe").info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BPE] Iter {merge_iterations}: merging pair {best_pair} (freq={max_freq}), vocab size={len(token_to_id)}")
        merged_bytes = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        if merged_bytes in token_to_id:
            pair_counter.add_skipped_pair(best_pair)
            continue
        token_to_id[merged_bytes] = next_id
        id_to_token[next_id] = merged_bytes
        merges.append((id_to_token[best_pair[0]], id_to_token[best_pair[1]]))
        merged_token_id = next_id
        next_id += 1
        affected_words = pair_counter.pair_to_words.get(best_pair, None)
        t_apply_merge = time.time()
        word_freqs_dict, total_merged, changed_words, affected_words_set = apply_merge_fast(
            word_freqs, best_pair, merged_token_id, affected_words
        )
        apply_merge_total_time = apply_merge_total_time + (time.time() - t_apply_merge) if 'apply_merge_total_time' in locals() else (time.time() - t_apply_merge)
        if total_merged == 0:
            pair_counter.add_skipped_pair(best_pair)
            continue
        for word in affected_words_set:
            word_freqs.pop(word, None)
        word_freqs.update(word_freqs_dict)
        affected_new_words = {w: f for w, f in word_freqs_dict.items() if merged_token_id in w}
        t_update = time.time()
        pair_counter.update_pairs(changed_words, affected_new_words)
        update_pairs_total_time += time.time() - t_update
        merge_iterations += 1
        if merge_bar:
            merge_bar.update(1)
    if merge_bar:
        merge_bar.close()
    timings['bpe_merge_loop'] = time.time() - t0
    timings['update_pairs_total'] = update_pairs_total_time
    timings['get_best_pair_total'] = get_best_pair_total_time
    timings['apply_merge_total'] = apply_merge_total_time if 'apply_merge_total_time' in locals() else 0.0
    logging.getLogger("toknix.bpe").info("\n===== Timing Report =====")
    for k, v in timings.items():
        logging.getLogger("toknix.bpe").info(f"{k:30s}: {v:.4f} sec")
    logging.getLogger("toknix.bpe").info(f"merge iterations{' ':14s}: {merge_iterations}")
    logging.getLogger("toknix.bpe").info("=========================")
    return id_to_token, merges
