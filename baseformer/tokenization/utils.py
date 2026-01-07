"""Utility functions for BPE Tokenization."""

from __future__ import annotations

import os
from typing import BinaryIO
from typing import Dict, List, Tuple, TYPE_CHECKING
import pathlib
from functools import lru_cache

if TYPE_CHECKING:
    from baseformer.tokenization.bpe import PairData, WordData


def merge_counts(dicts: List[Dict[int, int]]) -> Dict[int, int]:
    """Merges a list of dictionary of counts summing values across all dicts."""
    res = {}
    for d in dicts:
        for idx, count in d.items():
            res[idx] = res.get(idx, 0) + count
    return res


def find_counts(seq: List[int]) -> Dict[Tuple[int, int], int]:
    """Finds the number of occurrences of all pairs present in the sequence."""
    counts = {}
    for k1, k2 in zip(seq, seq[1:]):
        counts[(k1, k2)] = counts.get((k1, k2), 0) + 1
    return counts


def apply_merge(seq: List[int],
                pair: Tuple[int, int],
                new_token_idx: int
                ) -> Tuple[List[int], List[Tuple[int]]]:
    """Substitutes all occurences of the pair in the input sequence with the new token idx.
    If the pair is not present then returns seq unchanged
    
    Returns:
        new_sequence - List with the new token substituted in
        
        positive_pairs - List of (potentially repeating), new adjacent pairs introduced.
                         Pairs are uniquely identified by the left most index."""
    new_seq = []
    positive_pairs =[]
    i = 0
    new_seq_idx = -1
    while i < len(seq):
        # If we are not at the last element and we found the pair
        if (i < len(seq) - 1) and (seq[i] == pair[0]) and (seq[i+1] == pair[1]):
            new_seq.append(new_token_idx)
            i += 2  # Skip both elements of the pair
        else:
            new_seq.append(seq[i])
            i += 1
        new_seq_idx += 1

        # Add positive pairs
        if new_seq_idx > 0:
            if ((new_seq[new_seq_idx - 1] == new_token_idx)
                 or (new_seq[new_seq_idx] == new_token_idx)):
                 positive_pairs.append((new_seq[new_seq_idx - 1], new_seq[new_seq_idx]))

    return new_seq, positive_pairs


def compute_deltas(positive_pairs: List[Tuple[int, int]],
                   negative_pairs: List[Tuple[int, int]],
                   word_data: WordData
                   ) -> Dict[Tuple[int, int], PairData]:
    """Finds the alterations that should be made to the PairData map given a list of adjacent
    pairs to the one being substituted and the new token id.
    
    Arguments:
        positive_pairs - All occurences of all pairs introduced by the presence of the 
                         new token.
        negative_pairs - All occurences of all pairs removed by the presence of the 
                         new token.
        new_token_id - Id of the new token to compute which are the novel pairs.
        
        word_data - Current word being updated.

    Returns:
        deltas - For each pair returns a pair data object with the freq indicating
                 the total changes induced by the new token in this word.
    """
    from baseformer.tokenization.bpe import PairData

    deltas = {}

    def update(adj_pair, count):
        if not adj_pair in deltas:
            deltas[adj_pair] = PairData(0, set([word_data.idx])) # We will preserve set at other point
        deltas[adj_pair].freq += count
        return deltas

    for p in positive_pairs:
        update(p, word_data.count)
    
    for p in negative_pairs:
        update(p, -word_data.count)

    return deltas


def find_negative_pairs(seq: List[int], target: int) -> List[Tuple[int, int]]:
    """Finds all occurence of pairs adjacent to the given target pair.

    Returns:
        negative_pairs - Lists of pairs that overlap to the
                                left (resp right) of the target pair
    For example:
        target: (2, 3) - seq: [1, 2, 3, 4] would return List[(1,2)] List[(3,4)]

    If the target appears several times in the sequence then we return all instances

    For example:
        target: (2, 3) - seq: [1, 2, 3, 1, 2, 3] would return List[(1,2), (3,1)], List[(1, 2)]
    """
    if len(seq) == 0:
        raise ValueError("Received empty sequence of tokens.")
    negative_pairs = []
    pairs = zip(seq, seq[1:])
    prev_pair = None
    prev_prev_pair = None
    for i, p in enumerate(pairs):
         # Left check
         # In the case of repeating pairs [2, 3, 2, 3]
         # the pair at index 0 registers (3, 2).
         # So it shouldn't be counted from the left by the pair
         # starting at index 2.
        if (i > 0) and (p == target) and (p != prev_prev_pair):
            negative_pairs.append((seq[i - 1], seq[i]))
        if (i < len(seq) - 2) and (p == target): # Right check
            negative_pairs.append((seq[i + 1], seq[i + 2]))
        prev_prev_pair = prev_pair
        prev_pair = p
    return negative_pairs


def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    return s


# TODO: this can be seriously optimized
# We iterate over the entire string for each token and create intermediary strings
def remove_special_tokens(text: str, special_tokens: List[str]) -> str: 
    result = text
    for token in special_tokens:
        result = result.replace(token, '')
    return result



def find_chunk_boundaries(
    file: BinaryIO,
    split_special_token: bytes,
    chunk_size: int = 10 * 1024 * 1024 # 10MB
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.

    If the file is smaller than chunk_size, then return a single chunk.

    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    num_chunks = file_size // chunk_size if file_size > chunk_size else 1

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d
