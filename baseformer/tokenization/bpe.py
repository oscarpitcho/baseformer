"""
Logic to Tokenize using the BPE algorithm as implemented in GPT2.
Supports training from scratch, encoding, decoding, and introduction of special
tokens defined after tokenizer trainging.

Implementation Outline:
- The Pair Map: Stores and frequency of each pair and words where it occurs
- The Word Map: Stores the # of occurence of each word and the current encoding
- The Max Heap: Find in O(1) time the highest frequency pair. Can be stale and is checked against pair map.

This approach is inspired from Andrej Karpathy's Min-BPE Rust Implementation:

https://github.com/karpathy/minbpe/tree/master
"""

import heapq
import logging
from typing import Any, List, Tuple, Set, Dict, Iterable, Iterator
import regex as re
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import chain, repeat
from baseformer.tokenization.utils import find_chunk_boundaries, gpt2_bytes_to_unicode
from baseformer.tokenization.utils import compute_deltas, find_counts, find_negative_pairs, apply_merge
from dataclasses import dataclass


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
logger = logging.getLogger(__name__)

@dataclass(frozen=True, order=True)
class MergeJob:
    """Represents a pair candidate for merging. Internally stores the frequency
    as a negative number as python heaps are min-only. Should be built
    using the build method, and only freq should be accessed."""
    _freq_internal: int
    pair: Tuple[int, int] 

    @property
    def freq(self) -> int:
        return -self._freq_internal

    @classmethod
    def build(cls, freq: int, pair: Tuple[int, int]):
        return cls(-freq, pair)

@dataclass
class WordData:
    count: int # Total occurences in the corpus
    encoding: List[int]
    idx: int 

@dataclass
class PairData:
    freq: int
    locations: Set[int] # Frozen after generation

class BPETokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] | None = None):
        """Default special tokens are <|endoftext|>, additional special tokens can be added."""
        self.vocab = vocab
        self.merges = merges
        self.merges_dict = dict(zip(merges, range(len(merges))))
        self.vocab_reverse_map = {bytes_seq: token_id for token_id, bytes_seq in self.vocab.items()}
        self.special_tokens = set()
        if special_tokens is not None:
            self.special_tokens.update(special_tokens)
        self.special_tokens.add("<|endoftext|>")
        self.special_tok_pat = "(" + '|'.join(re.escape(token) for token in self.special_tokens) + ")"

        self.special_tokens_reverse_map: Dict[bytes, int] = {}

        # Special token handling
        i = len(self.vocab)
        for st in self.special_tokens:
                self.special_tokens_reverse_map[st.encode('utf-8')] = len(self.vocab)
                self.vocab[i] = st.encode('utf-8')
                i += 1
        
    def encode(self, text: str) -> List[int]:
        if self.special_tok_pat is None:
            return self._encode_segment(text)
        else:
            res = []
            segments = re.splititer(self.special_tok_pat, text)
            for i, seq in enumerate(segments):
                if i % 2 == 1: # Special Token
                    st_bytes = seq.encode("utf-8")
                    if not (st_bytes in self.special_tokens_reverse_map):
                        logger.warning(f"Unknown special token {seq} found, encoding as normal word.")
                        res.extend(self._encode_word(st_bytes))
                    else:
                        res.append(self.special_tokens_reverse_map[st_bytes]) 
                else: # Normal Text
                    res.extend(self._encode_segment(seq))
        return res


    def _encode_segment(self, segment: str) -> List[int]:
        """Encode regular text segment (no special tokens), pretokenizes then applies BPE encoding."""
        tokens = []
        matches = re.finditer(PAT, segment)
        for match in matches:
            word_bytes = match.group().encode("utf-8")
            tokens.extend(self._encode_word(word_bytes))
        return tokens


    def _encode_word(self, w_bytes: bytes) -> List[int]:
        """Encodes a sequece of bytes into the corresponding sequence of token IDs"""
        current_encoding = list(w_bytes)
        while True:
            pairs = find_counts(current_encoding)
            pairs_merge_idx = {}

            # Check if the pair is a new token using the merges dict.
            # Keep the merge idx to prioritize which pair to merge.
            for id_p in pairs:
                bytes_p = (self.vocab[id_p[0]], self.vocab[id_p[1]])
                if bytes_p in self.merges_dict:
                    pairs_merge_idx[bytes_p] = self.merges_dict[bytes_p]

            if len(pairs_merge_idx) == 0:
                break

            # Pair to merge is the one with lowest index.
            # This prioritizes showing high-frequency tokens to the model
            # over larger less frequent tokens.
            min_pair, _ = min(pairs_merge_idx.items(), key=lambda x: x[1])
            min_pair_bytes = min_pair[0] + min_pair[1]
            min_pair_token_id = self.vocab_reverse_map[min_pair_bytes]
            l_elemt_token_id = self.vocab_reverse_map[min_pair[0]]
            r_elemt_token_id = self.vocab_reverse_map[min_pair[1]]
            min_pair_token_ids = (l_elemt_token_id, r_elemt_token_id)
            current_encoding, _ = apply_merge(current_encoding, min_pair_token_ids, min_pair_token_id)

        return current_encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings into a stream of token IDs."""
        for element in iterable:
            for token_id in self.encode(element):
                yield token_id


    def decode(self, ids: List[int]) -> str:
        return b''.join(self.vocab[id] for id in ids).decode('utf-8')

    def decode_debug(self, ids: List[int]) -> List[str]:
        return [self.vocab[id].decode('utf-8') for id in ids]


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as vf, open(merges_filepath, "rb") as mf:
            vocab = pickle.load(vf)
            merges = pickle.load(mf)
        return cls(vocab, merges, special_tokens)

    @staticmethod
    def prettify_vocab(vocab: dict[int, bytes]) -> dict[str, str]:
        """Converts the vocabulary from bytes (with non json-serializable bytes) to printable characters."""
        enc = gpt2_bytes_to_unicode()        # byte -> printable char
        return {str(tok_id): ''.join(enc[b] for b in token_bytes)
                for tok_id, token_bytes in vocab.items()}

    @staticmethod
    def prettify_merges(merges: list[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
        """Converts the merges from bytes (with non json-serializable bytes) to printable characters."""
        enc = gpt2_bytes_to_unicode()
        def pretty_bytes(b: bytes) -> str:
            return ''.join(enc[x] for x in b)
        return [(pretty_bytes(left), pretty_bytes(right)) for left, right in merges]
            

def _pretokenize_chunk(input_path: str, start: int, end: int, special_tok_pat: str) -> List[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    chunk_seq = []
    
    # Ensure no tokenization across special tokens.
    # Iterators for minimal memory usage.
    for seg in re.splititer(special_tok_pat, chunk):
        for w in re.finditer(PAT, seg):
            chunk_seq.append(w.group())

    del chunk
    return chunk_seq    

def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str], n_processes: int = 1):

    """
    Train a byte pair encoding (BPE) tokenizer on a text file.

    Args:
        input_path: Path to UTF-8 text containing training data.
        vocab_size: Target vocabulary size.
        special_tokens: Special tokens which should not be tokenized; expected format `<|TOKEN|>`.
        n_processes: Number of processes to use for preprocessing.
    Returns:
        vocab: Mapping from token id to token bytes. Does not include special tokens.
        merges: Ordered list of merge pairs `(left_bytes, right_bytes)` in creation order.
    """

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    logger.info("Reading training data from %s", input_path)

    # Init data structures
    merges: List[Tuple[bytes, bytes]] = []
    idx_to_word: Dict[int, str] = {}
    word_map: Dict[str, WordData] = {}
    pairs_map: Dict[Tuple[int, int], PairData] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    pairs_heap: List[MergeJob] = []

    special_tok_pat = '|'.join(re.escape(token) for token in special_tokens)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, b"<|endoftext|>")
        logger.info("Found %d chunk boundaries", len(boundaries))

        # ------- Preprocessing -------
        raw_word_sequence = []
        if n_processes > 1:
            logger.info("Using %d processes for preprocessing", n_processes)
            fork_ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=n_processes, mp_context=fork_ctx) as executor:
                results = executor.map(
                    _pretokenize_chunk,
                    repeat(input_path),
                    boundaries[:-1],
                    boundaries[1:],
                    repeat(special_tok_pat),
                )
        
            # Efficiently flatten the iterator of lists using itertools.chain
            raw_word_sequence = list(chain.from_iterable(results))
        else:
            logger.info("Using 1 process for preprocessing")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                raw_word_sequence.extend(_pretokenize_chunk(input_path, start, end, special_tok_pat))

        logger.info("Found %d words in the corpus", len(raw_word_sequence))
        _initialize_word_map(word_map, idx_to_word, raw_word_sequence)
        logger.info("Initialized %d distinct words", len(word_map))
        _initialize_pairs_map(pairs_map, word_map)
        logger.info("Initialized %d distinct pairs", len(pairs_map))

        # Feeding all pairs into a list and creating heap
        for pair, data in pairs_map.items():
            merge_job = MergeJob.build(data.freq, pair)
            pairs_heap.append(merge_job)
        heapq.heapify(pairs_heap)

    # ------ Merging ------
    actual_vocab_size = vocab_size - len(special_tokens)
    _ = _run_merges(pairs_heap, pairs_map, merges, word_map, vocab, idx_to_word, actual_vocab_size)

    logger.info("Training finished with %d merges and final vocab size %d", len(merges), len(vocab))

    return vocab, merges


def _initialize_word_map(word_map: Dict[str, WordData],
                         idx_to_word: Dict[int, str],
                         word_sequence: List[str]) -> None:
    for word in word_sequence:
        encoding = list(word.encode("utf-8")) # each element is an int 0-255
        if word not in word_map:
            word_map[word] = WordData(1, encoding, len(idx_to_word))
            idx_to_word[len(idx_to_word)] = word # This makes it hard to parallelize.
        else:
            word_map[word].count += 1

def _initialize_pairs_map(pairs_map: Dict[Tuple[int, int], PairData],
                          word_map: Dict[str, WordData]) -> None:
    for word, data in word_map.items():
        pairs = find_counts(data.encoding)
        for p, freq in pairs.items():
            if p not in pairs_map:
                pairs_map[p] = PairData(freq, set([data.idx]))
            else:
                pairs_map[p].freq += (freq * data.count)
                pairs_map[p].locations.add(data.idx)

def _run_merges(pairs_heap: List[MergeJob],
                pairs_map: Dict[Tuple[int, int], PairData],
                merges: List[Tuple[bytes, bytes]],
                word_map: Dict[str, WordData],
                vocab: Dict[int, bytes],
                idx_to_word: Dict[int, str],
                vocab_size: int) -> int:
    """Compute the merging process."""

    current_index = 256
    # ------ Merging ------
    # Something to be merged and we have room to expand vocab
    while len(pairs_heap) > 0 and current_index < vocab_size:

        merge_job = heapq.heappop(pairs_heap)
        pair = merge_job.pair
        freq = merge_job.freq

        # Node is stale, adjust, push and try again
        if pairs_map[pair].freq != freq: 
            if pairs_map[pair].freq > 0:
                merge_job_updated = MergeJob.build(pairs_map[pair].freq, pair)
                heapq.heappush(pairs_heap, merge_job_updated)
            continue

        if (current_index % 1000) == 0:
            logger.info("Running merge: %d of %d", current_index, vocab_size)

        # Debug, check if we just added something already in the vocab
        if vocab[pair[0]] + vocab[pair[1]] in vocab.values():
            logger.warning("Pair %s already in vocab, heap freq: %d, pair freq: %d", vocab[pair[0]] + vocab[pair[1]], pairs_heap[0].freq, pairs_map[pair].freq)
            raise ValueError("Pair already in vocab")

        # Create new token and update data structures:
        new_token_id = current_index
        current_index += 1
        vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]
        pairs_map[pair].freq = 0 # This pair is merged
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        locations = pairs_map[pair].locations

        deltas = []
        for idx in locations:
            word = idx_to_word[idx]
            word_data = word_map[word]

            # Find old and new pairs affected
            negative_pairs = find_negative_pairs(word_data.encoding, pair)
            new_encoding, positive_pairs = apply_merge(word_data.encoding, pair, new_token_id)
            deltas.append(compute_deltas(positive_pairs, negative_pairs, word_data))
            word_map[word].encoding = new_encoding

        # Merge all the deltas into one via sum and union reduction
        total_delta = {} # How pairs are affected by merge
        for delta in deltas:
            for pair, p_data in delta.items():
                if pair not in total_delta:
                    total_delta[pair] = p_data
                else:
                    total_delta[pair].freq += p_data.freq
                    total_delta[pair].locations |= p_data.locations

        # Update frequency map and push new nodes in heap
        for pair, p_data in total_delta.items():
            if pair not in pairs_map: # New pair from merge
                pairs_map[pair] = p_data
            else: 
                pairs_map[pair].freq += p_data.freq
                # Pair no longer appears, no merging jobs
                if pairs_map[pair].freq <= 0:
                    continue
               
            new_node = MergeJob.build(pairs_map[pair].freq, pair)
            heapq.heappush(pairs_heap, new_node)

    return current_index



