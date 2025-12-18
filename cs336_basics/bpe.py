from typing import List, Tuple, Dict
from pretokenization_example import find_chunk_boundaries


NUM_PROCESSES = 4

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index
    special_tokens: List[str] | None = None # List of special tokens to add to the vocabulary.



class BPETokenizer:

    def __init__(params: BPETokenizerParams):
        self.params = params
    

    def encode
        


    def decode():
        pass


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):


        return cls(params)


def merge_counts(counts1: Dict[bytes, int], counts2: Dict[bytes, int]) -> Dict[bytes, int]:
    """Merges two dictionary of counts summing keys present in both."""
    merged = counts1.copy()
    for k, count in counts2.items():
        mergerd[k] = merged.get_or_else(k, 0) + count


def train_bpe_basic(input_path: str, input_size: int, vocab_size: int, special_tokens: List[str]):


    pass

def train_bpe(input_path: str, input_size: int, vocab_size: int, special_tokens: List[str]):

    """

    input_path: str Path to a text file with BPE tokenizer training data.
    vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
    special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
    otherwise affect BPE training.
    Your BPE training function should return the resulting vocabulary and merges:
    vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
    lary) to bytes (token bytes).
    merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
    is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
    <token2>. The merges should be ordered by order of creation.
    
    """

    with open(input_path, "rb") as f:
        f = f.readlines()


    boundaries = find_chunk_boundaries(f, NUM_PROCESSES = 4, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token



    return vocab, merges