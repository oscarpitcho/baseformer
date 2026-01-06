import heapq
from collections import Counter
import pytest
from cs336_basics.tokenization.utils import (
    merge_counts,
    find_counts,
    apply_merge,
    find_negative_pairs,
    compute_deltas,

)
from cs336_basics.tokenization.bpe import (
    WordData,
    PairData,
    BPETokenizer,
    _run_merges,
    MergeJob,
    _initialize_word_map,
    _initialize_pairs_map,
)

class TestComputeDeltas:
    """Tests for compute_deltas helper when updating pair frequencies."""

    def test_compute_deltas_left_and_right_pairs(self):
        """Updates remove old neighbors and add new ones on both sides."""
        word = WordData(count=2, encoding=[1, 2, 3, 4], idx=7)
        positive = [(1, 5), (5, 4)]
        negative = [(1, 2), (3, 4)]
        deltas = compute_deltas(positive_pairs=positive, negative_pairs=negative, word_data=word)

        assert deltas[(1, 2)] == PairData(freq=-2, locations={7})
        assert deltas[(1, 5)] == PairData(freq=2, locations={7})
        assert deltas[(3, 4)] == PairData(freq=-2, locations={7})
        assert deltas[(5, 4)] == PairData(freq=2, locations={7})


    def test_compute_deltas_accumulates_duplicates(self):
        """Multiple occurrences of the same neighbor accumulate counts."""
        word = WordData(count=1, encoding=[9, 8, 9, 8], idx=0)
        positive = [(9, 10)]
        negative = [(9, 8), (9, 8)]
        deltas = compute_deltas(positive_pairs=positive, negative_pairs=negative, word_data=word)

        assert deltas[(9, 8)] == PairData(freq=-2, locations={0})
        assert deltas[(9, 10)] == PairData(freq=1, locations={0})


    def test_compute_deltas_only_right_pairs(self):
        """Handles only right-side neighbors."""
        word = WordData(count=3, encoding=[4, 4, 4], idx=1)
        positive = [(6, 4)]
        negative = [(4, 4)]
        deltas = compute_deltas(positive_pairs=positive, negative_pairs=negative, word_data=word)

        assert deltas[(4, 4)] == PairData(freq=-3, locations={1})
        assert deltas[(6, 4)] == PairData(freq=3, locations={1})


    def test_compute_deltas_no_neighbors_returns_empty(self):
        """Returns empty deltas when there are no adjacent pairs."""
        word = WordData(count=5, encoding=[1], idx=2)
        deltas = compute_deltas(positive_pairs=[], negative_pairs=[], word_data=word)

        assert deltas == {}


class TestMergeCounts:
    def test_empty_list(self):
        """Test merging an empty list of dictionaries."""
        result = merge_counts([])
        assert result == {}

    def test_single_dict(self):
        """Test merging a single dictionary."""
        dicts = [{1: 5, 2: 3}]
        result = merge_counts(dicts)
        assert result == {1: 5, 2: 3}

    def test_multiple_dicts_no_overlap(self):
        """Test merging multiple dictionaries with no overlapping keys."""
        dicts = [{1: 5, 2: 3}, {3: 7, 4: 2}]
        result = merge_counts(dicts)
        assert result == {1: 5, 2: 3, 3: 7, 4: 2}

    def test_multiple_dicts_with_overlap(self):
        """Test merging multiple dictionaries with overlapping keys."""
        dicts = [{1: 5, 2: 3}, {1: 2, 3: 7}]
        result = merge_counts(dicts)
        assert result == {1: 7, 2: 3, 3: 7}

    def test_multiple_dicts_all_overlap(self):
        """Test merging multiple dictionaries where all keys overlap."""
        dicts = [{1: 5, 2: 3}, {1: 2, 2: 4}, {1: 1, 2: 1}]
        result = merge_counts(dicts)
        assert result == {1: 8, 2: 8}

    def test_empty_dicts_in_list(self):
        """Test merging a list containing empty dictionaries."""
        dicts = [{}, {1: 5}, {}]
        result = merge_counts(dicts)
        assert result == {1: 5}


class TestFindCounts:
    def test_empty_sequence(self):
        """Test finding counts in an empty sequence."""
        result = find_counts([])
        assert result == {}

    def test_single_element(self):
        """Test finding counts in a sequence with a single element."""
        result = find_counts([1])
        assert result == {}

    def test_two_elements(self):
        """Test finding counts in a sequence with two elements."""
        result = find_counts([1, 2])
        assert result == {(1, 2): 1}

    def test_multiple_elements_no_repeats(self):
        """Test finding counts in a sequence with no repeated pairs."""
        result = find_counts([1, 2, 3, 4])
        assert result == {(1, 2): 1, (2, 3): 1, (3, 4): 1}

    def test_multiple_elements_with_repeats(self):
        """Test finding counts in a sequence with repeated pairs."""
        result = find_counts([1, 2, 1, 2, 3, 1, 2])
        assert result == {(1, 2): 3, (2, 1): 1, (2, 3): 1, (3, 1): 1}


    def test_same_element_repeated(self):
        """Test finding counts when the same element is repeated."""
        result = find_counts([1, 1, 1, 1])
        assert result == {(1, 1): 3}

class TestApplyMerge:
    def test_empty_sequence(self):
        """Test applying merge to an empty sequence."""
        new_seq, positives = apply_merge([], (1, 2), 10)
        assert new_seq == []
        assert positives == []

    def test_no_match(self):
        """Test applying merge when the pair doesn't appear."""
        new_seq, positives = apply_merge([1, 2, 3, 4], (5, 6), 10)
        assert new_seq == [1, 2, 3, 4]
        assert positives == []

    def test_single_match(self):
        """Test applying merge when the pair appears once."""
        new_seq, positives = apply_merge([1, 2, 3, 4], (2, 3), 10)
        assert new_seq == [1, 10, 4]
        assert positives == [(1, 10), (10, 4)]

    def test_multiple_matches(self):
        """Test applying merge when the pair appears multiple times."""
        new_seq, positives = apply_merge([1, 2, 3, 2, 3, 4], (2, 3), 10)
        assert new_seq == [1, 10, 10, 4]
        assert positives == [(1, 10), (10, 10), (10, 4)]

    def test_match_at_start(self):
        """Test applying merge when the pair appears at the start."""
        new_seq, positives = apply_merge([1, 2, 3, 4], (1, 2), 10)
        assert new_seq == [10, 3, 4]
        assert positives == [(10, 3)]

    def test_match_at_end(self):
        """Test applying merge when the pair appears at the end."""
        new_seq, positives = apply_merge([1, 2, 3, 4], (3, 4), 10)
        assert new_seq == [1, 2, 10]
        assert positives == [(2, 10)]

    def test_overlapping_pairs(self):
        """Test applying merge when pairs overlap (should not merge overlapping)."""
        new_seq, positives = apply_merge([1, 2, 2, 3], (2, 2), 10)
        assert new_seq == [1, 10, 3]
        assert positives == [(1, 10), (10, 3)]

    def test_consecutive_matches(self):
        """Test applying merge when matches appear consecutively."""
        new_seq, positives = apply_merge([1, 2, 3, 2, 3, 2, 3], (2, 3), 10)
        assert new_seq == [1, 10, 10, 10]
        assert positives == [(1, 10), (10, 10), (10, 10)]

    def test_single_element_sequence(self):
        """Test applying merge to a sequence with a single element."""
        new_seq, positives = apply_merge([1], (1, 2), 10)
        assert new_seq == [1]
        assert positives == []


class TestFindNegativePairs:
    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            find_negative_pairs([], (1, 2))

    def test_target_in_middle_returns_left_and_self(self):
        seq = [1, 2, 3, 4]
        target = (2, 3)
        negatives = find_negative_pairs(seq, target)
        assert negatives == [(1, 2), (3, 4)]

    def test_target_at_start(self):
        seq = [1, 2, 3]
        target = (1, 2)
        negatives = find_negative_pairs(seq, target)
        assert negatives == [(2, 3)]

    def test_target_at_end(self):
        seq = [1, 2, 3]
        target = (2, 3)
        negatives = find_negative_pairs(seq, target)
        assert negatives == [(1, 2)]

    def test_multiple_occurrences(self):
        seq = [1, 2, 3, 1, 2, 3]
        target = (2, 3)
        negatives = find_negative_pairs(seq, target)
        assert negatives == [(1, 2), (3, 1), (1, 2)]

    def test_repeats(self):
        seq = [1, 2, 3, 2, 3]
        target = (2, 3)
        expected_negatives = [(1, 2), (3, 2)] 
        negatives = find_negative_pairs(seq, target)
        assert negatives == expected_negatives, f"Found {negatives}, expected {expected_negatives}" 


    def test_same_repeats_3(self):
        seq = [2, 1, 1, 2, 1, 1, 2]
        target = (1, 1)
        expected_negatives = [(2, 1), (1, 2), (2, 1), (1, 2)] 
        negatives = find_negative_pairs(seq, target)
        assert negatives == expected_negatives, f"Found {negatives}, expected {expected_negatives}" 
    
    def test_target_not_present_returns_empty(self):
        seq = [1, 2, 3]
        target = (3, 4)
        negatives = find_negative_pairs(seq, target)
        assert negatives == []


class TestBPE:
    """Tests for BPETokenizer encode/decode methods."""

    def test_encode_decode_empty_string(self):
        """Test encode/decode round-trip with empty string."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        tokenizer = BPETokenizer(vocab, merges)
        text = ""
        ids = tokenizer.encode(text)
        expected_ids = []
        print(f"Expected: {expected_ids}, Got: {ids}")
        assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"
        decoded = tokenizer.decode(ids)
        print(f"Expected decoded: {text!r}, Got: {decoded!r}")
        assert decoded == text, f"Expected {text!r}, got {decoded!r}"


    def test_encode_decode_simple_text(self):
        """Test encode/decode round-trip with simple text."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        tokenizer = BPETokenizer(vocab, merges)
        text = "hello world"
        ids = tokenizer.encode(text)
        # With no merges, should encode as individual bytes
        expected_ids = [ord(c) for c in text]
        print(f"Expected: {expected_ids}, Got: {ids}")
        assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"
        decoded = tokenizer.decode(ids)
        print(f"Expected decoded: {text!r}, Got: {decoded!r}")
        assert decoded == text, f"Expected {decoded!r}, got {text!r}"

    def test_encode_decode_with_merged_tokens(self):
        """Test encode/decode with merged tokens using lowest rank merge first."""
        vocab = {i: bytes([i]) for i in range(256)}
        # Create merges: (h, e) -> 256, (he, l) -> 257, (hel, l) -> 258
        # Lower rank = earlier merge = higher priority
        merges = [
            (b'h', b'e'),      # rank 0 -> token 256
            (b'he', b'l'),     # rank 1 -> token 257
            (b'hel', b'l'),    # rank 2 -> token 258
        ]
        # Build vocab with merged tokens
        vocab[256] = b'he'
        vocab[257] = b'hel'
        vocab[258] = b'hell'
        tokenizer = BPETokenizer(vocab, merges)
        text = "hello"
        ids = tokenizer.encode(text)
        # BPE algorithm: start with [h, e, l, l, o]
        # First merge: (h, e) at rank 0 -> [256, l, l, o] (he, l, l, o)
        # Second merge: (he, l) at rank 1 -> [257, l, o] (hel, l, o)
        # Third merge: (hel, l) at rank 2 -> [258, o] (hell, o)
        # No more merges possible
        expected_ids = [258, ord('o')]
        print(f"Expected: {expected_ids}, Got: {ids}")
        assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"
        decoded = tokenizer.decode(ids)
        print(f"Expected decoded: {text!r}, Got: {decoded!r}")
        assert decoded == text, f"Expected {text!r}, got {decoded!r}"

    def test_encode_decode_special_characters_unicode(self):
        """Test encode/decode round-trip with special characters and Unicode."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        tokenizer = BPETokenizer(vocab, merges)
        test_cases = [
            "hello! world?",
            "café",
            "   ",
            "test\nnewline\ttab",
        ]
        for text in test_cases:
            ids = tokenizer.encode(text)
            # With no merges, should encode as individual bytes
            expected_ids = list(text.encode("utf-8"))
            print(f"Text: {text!r}")
            print(f"Expected: {expected_ids}, Got: {ids}")
            assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"
            decoded = tokenizer.decode(ids)
            print(f"Expected decoded: {text!r}, Got: {decoded!r}")
            assert decoded == text, f"Expected {text!r}, got {decoded!r}"

    def test_encode_decode_partial_match_fallback(self):
        """Test encode/decode when partial matches require fallback to bytes."""
        vocab = {i: bytes([i]) for i in range(256)}
        # Create merge: (x, y) -> 256
        merges = [(b'x', b'y')]  # rank 0 -> token 256
        vocab[256] = b'xy'
        tokenizer = BPETokenizer(vocab, merges)
        text = "xyz"
        ids = tokenizer.encode(text)
        # BPE algorithm: start with [x, y, z]
        # First merge: (x, y) at rank 0 -> [256, z] (xy, z)
        # No more merges possible
        expected_ids = [256, ord('z')]
        print(f"Expected: {expected_ids}, Got: {ids}")
        assert ids == expected_ids, f"Expected {expected_ids}, got {ids}"
        decoded = tokenizer.decode(ids)
        print(f"Expected decoded: {text!r}, Got: {decoded!r}")
        assert decoded == text, f"Expected {text!r}, got {decoded!r}"


class TestRunMerges:
    def test_run_merges_does_not_remerge_exhausted_pair(self):
        """Ensure a pair with zero remaining frequency is not merged again."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        word_map: dict[str, WordData] = {}
        idx_to_word: dict[int, str] = {}
        _initialize_word_map(word_map, idx_to_word, ["aaa"])

        pairs_map: dict[tuple[int, int], PairData] = {}
        _initialize_pairs_map(pairs_map, word_map)

        pairs_heap = [MergeJob.build(data.freq, pair) for pair, data in pairs_map.items()]
        heapq.heapify(pairs_heap)

        _run_merges(
            pairs_heap=pairs_heap,
            pairs_map=pairs_map,
            merges=merges,
            word_map=word_map,
            vocab=vocab,
            idx_to_word=idx_to_word,
            vocab_size=260,  # large enough to expose re-merge if it happens
        )

        assert Counter(merges)[(b"a", b"a")] == 1, "Pair (a, a) should be merged once"

    def test_run_merges_noop_on_empty_corpus(self):
        """Empty corpus should produce no merges and leave structures unchanged."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges: list[tuple[bytes, bytes]] = []

        word_map: dict[str, WordData] = {}
        idx_to_word: dict[int, str] = {}
        _initialize_word_map(word_map, idx_to_word, [""])

        pairs_map: dict[tuple[int, int], PairData] = {}
        _initialize_pairs_map(pairs_map, word_map)
        pairs_heap = [MergeJob.build(data.freq, pair) for pair, data in pairs_map.items()]

        _run_merges(
            pairs_heap=pairs_heap,
            pairs_map=pairs_map,
            merges=merges,
            word_map=word_map,
            vocab=vocab,
            idx_to_word=idx_to_word,
            vocab_size=258,
        )

        assert merges == [], "No merges should be produced for empty input"
        assert pairs_map == {}, "No pairs should exist for empty input"
        assert word_map == {"": WordData(count=1, encoding=[], idx=0)}

    def test_run_merges_does_not_merge_pairs_removed_by_prior_merge(self):
        """Pairs eliminated by another merge should not reappear later."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        word_map: dict[str, WordData] = {}
        idx_to_word: dict[int, str] = {}
        _initialize_word_map(word_map, idx_to_word, ["abc", "xbc"])

        pairs_map: dict[tuple[int, int], PairData] = {}
        _initialize_pairs_map(pairs_map, word_map)

        pairs_heap = [MergeJob.build(data.freq, pair) for pair, data in pairs_map.items()]
        heapq.heapify(pairs_heap)

        _run_merges(
            pairs_heap=pairs_heap,
            pairs_map=pairs_map,
            merges=merges,
            word_map=word_map,
            vocab=vocab,
            idx_to_word=idx_to_word,
            vocab_size=260,  # enough room for any stale merges to surface
        )

        expected_merges = [
            (b"b", b"c"),   # merge highest-frequency pair first
            (b"a", b"bc"),
            (b"x", b"bc"),
        ]

        assert merges == expected_merges, f"Unexpected merges: {merges}"


    def test_run_merges_repeats(self):
        """Pairs eliminated by another merge should not reappear later."""
        vocab = {i: bytes([i]) for i in range(256)}
        merges = []
        word_map: dict[str, WordData] = {}
        idx_to_word: dict[int, str] = {}
        words = [" aing", " bing"]
        _initialize_word_map(word_map, idx_to_word, words)

        pairs_map: dict[tuple[int, int], PairData] = {}
        _initialize_pairs_map(pairs_map, word_map)

        pairs_heap = [MergeJob.build(data.freq, pair) for pair, data in pairs_map.items()]
        heapq.heapify(pairs_heap)

        _run_merges(
            pairs_heap=pairs_heap,
            pairs_map=pairs_map,
            merges=merges,
            word_map=word_map,
            vocab=vocab,
            idx_to_word=idx_to_word,
            vocab_size=10000,  # enough room for any stale merges to surface
        )

        expected_merges = [
            (b"i", b"n"),   # merge highest-frequency pair first
            (b"in", b"g"),
            (b" ", b"a"),
            (b" ", b"b"),
            (b" a", b"ing"),
            (b" b", b"ing"),
        ]

        assert merges == expected_merges, f"Unexpected merges: {merges}"



class TestInitializeMaps:
    def test_initialize_word_map_counts_and_indices(self):
        """Adds new words, preserves order, and increments counts."""
        word_map: dict[str, WordData] = {}
        idx_to_word: dict[int, str] = {}
        words = ["a", "b", "a"]

        _initialize_word_map(word_map, idx_to_word, words)

        assert word_map["a"].count == 2
        assert word_map["b"].count == 1
        assert word_map["a"].encoding == [ord("a")]
        assert word_map["b"].encoding == [ord("b")]
        assert idx_to_word == {0: "a", 1: "b"}



    def test_initialize_pairs_map_handles_repeated_pair_same_word(self):
        """Repeated pairs in one word count multiple times but keep single location."""
        word_map = {
            "aaaa": WordData(count=1, encoding=[1, 1, 1, 1], idx=0),
        }
        pairs_map: dict[tuple[int, int], PairData] = {}

        _initialize_pairs_map(pairs_map, word_map)

        assert pairs_map[(1, 1)].freq == 3
        assert pairs_map[(1, 1)].locations == {0}
