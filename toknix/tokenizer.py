"""
Tokenizer class for byte-level BPE encoding/decoding.
"""
import regex as re
from typing import Dict, List, Tuple, Set, Iterator, Optional
import heapq

class Tokenizer:
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        import pickle
        with open(vocab_filepath, 'rb') as vf:
            vocab = pickle.load(vf)
        with open(merges_filepath, 'rb') as mf:
            merges = pickle.load(mf)
        if special_tokens:
            max_id = max(vocab.keys(), default=-1)
            for token in special_tokens:
                token_bytes = token.encode('utf-8')
                if token_bytes not in vocab.values():
                    max_id += 1
                    vocab[max_id] = token_bytes
        return cls(vocab, merges, special_tokens)

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        if self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_tokens.sort(key=len, reverse=True)
            self.special_token_pattern = re.compile(
                '|'.join(f'({token})' for token in escaped_tokens),
                re.DOTALL
            )
        else:
            self.special_token_pattern = None

    def encode(self, text: str) -> List[int]:
        byte_encoded = text.encode('utf-8')
        if self.special_tokens and byte_encoded:
            return self._encode_with_special_tokens(text)
        else:
            return self._bpe_encode(byte_encoded)

    def _encode_with_special_tokens(self, text: str) -> List[int]:
        tokens = []
        if self.special_token_pattern:
            special_matches = list(self.special_token_pattern.finditer(text))
            if not special_matches:
                return self._bpe_encode(text.encode('utf-8'))
            last_end = 0
            for match in special_matches:
                start, end = match.span()
                if start > last_end:
                    before_text = text[last_end:start]
                    tokens.extend(self._bpe_encode(before_text.encode('utf-8')))
                special_token = match.group(0)
                if special_token in self.special_tokens:
                    special_token_bytes = special_token.encode('utf-8')
                    if special_token_bytes in self.byte_to_id:
                        tokens.append(self.byte_to_id[special_token_bytes])
                    else:
                        tokens.extend(self._bpe_encode(special_token_bytes))
                last_end = end
            if last_end < len(text):
                tokens.extend(self._bpe_encode(text[last_end:].encode('utf-8')))
        return tokens

    def _bpe_encode(self, byte_encoded: bytes) -> List[int]:
        if not byte_encoded:
            return []
        tokens = [bytes([b]) for b in byte_encoded]
        n = len(tokens)
        prev = list(range(-1, n - 1))
        next = list(range(1, n + 1))
        next[-1] = -1
        alive = [True] * n
        pair_positions = {}
        heap = []
        def add_pair(pos: int):
            if pos == -1 or pos == n - 1:
                return
            if not (alive[pos] and alive[next[pos]]):
                return
            pair = (tokens[pos], tokens[next[pos]])
            if pair in self.merge_ranks:
                rank = self.merge_ranks[pair]
                heapq.heappush(heap, (rank, pos))
                pair_positions.setdefault(pair, set()).add(pos)
        for i in range(n - 1):
            add_pair(i)
        while heap:
            rank, pos = heapq.heappop(heap)
            if pos == -1 or pos == n - 1:
                continue
            if not alive[pos] or not alive[next[pos]]:
                continue
            pair = (tokens[pos], tokens[next[pos]])
            if pair not in self.merge_ranks or self.merge_ranks[pair] != rank:
                continue
            positions_to_merge = []
            if pair in pair_positions:
                candidates = list(pair_positions[pair])
            else:
                candidates = []
            for p in candidates:
                if p != -1 and p < n - 1 and alive[p] and alive[next[p]]:
                    current_pair = (tokens[p], tokens[next[p]])
                    if current_pair == pair:
                        positions_to_merge.append(p)
            if not positions_to_merge:
                continue
            positions_to_merge.sort()
            pair_positions[pair].difference_update(positions_to_merge)
            for left_pos in positions_to_merge:
                right_pos = next[left_pos]
                if not (alive[left_pos] and alive[right_pos]):
                    continue
                merged_token = tokens[left_pos] + tokens[right_pos]
                if merged_token not in self.byte_to_id:
                    continue
                tokens[left_pos] = merged_token
                alive[right_pos] = False
                nxt = next[right_pos]
                next[left_pos] = nxt
                if nxt != -1:
                    prev[nxt] = left_pos
                def remove_pair(pos_to_remove):
                    if pos_to_remove == -1 or pos_to_remove >= n - 1:
                        return
                    if not (alive[pos_to_remove] and alive[next[pos_to_remove]]):
                        return
                    p = (tokens[pos_to_remove], tokens[next[pos_to_remove]])
                    if p in pair_positions:
                        pair_positions[p].discard(pos_to_remove)
                remove_pair(prev[left_pos])
                remove_pair(left_pos)
                remove_pair(right_pos)
                add_pair(prev[left_pos])
                add_pair(left_pos)
        result = []
        i = 0
        while i != -1 and i < n:
            if alive[i]:
                result.append(tokens[i])
            i = next[i]
        return [self.byte_to_id[tok] for tok in result if tok in self.byte_to_id]

    def decode(self, token_ids: List[int]) -> str:
        byte_sequences = [self.vocab[token_id] for token_id in token_ids if token_id in self.vocab]
        if not byte_sequences:
            return ""
        return b''.join(byte_sequences).decode('utf-8', errors='replace')

    def encode_iterable(self, iterable: Iterator[str], show_progress: bool = False, total: Optional[int] = None, desc: str = "Encoding") -> Iterator[int]:
        """
        Encode an iterable of strings, optionally showing a progress bar.
        Args:
            iterable: Iterator of strings to encode.
            show_progress: If True, show a tqdm progress bar.
            total: Optional total number of items (for tqdm).
            desc: Description for the progress bar.
        Yields:
            int: Token IDs from all encoded chunks.
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterable, total=total, desc=desc)
            except ImportError:
                iterator = iterable
        else:
            iterator = iterable
        for chunk in iterator:
            for token_id in self.encode(chunk):
                yield token_id
