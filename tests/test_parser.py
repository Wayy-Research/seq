"""Tests for sequence parser."""

import pytest
from seq.parser import parse_sequence, format_sequence


class TestParseSequence:
    """Tests for parse_sequence function."""

    def test_simple_sequence(self):
        """Test parsing a simple comma-separated sequence."""
        seq = parse_sequence("1, 2, 3, 4, 5")
        assert seq.values == [1, 2, 3, 4, 5]
        assert seq.missing_indices == []

    def test_sequence_with_missing(self):
        """Test parsing sequence with ? for missing values."""
        seq = parse_sequence("1, 2, ?, 4, 5")
        assert seq.values == [1, 2, None, 4, 5]
        assert seq.missing_indices == [2]

    def test_multiple_missing(self):
        """Test sequence with multiple missing values."""
        seq = parse_sequence("?, 2, ?, 4, ?")
        assert seq.values == [None, 2, None, 4, None]
        assert seq.missing_indices == [0, 2, 4]

    def test_fibonacci_example(self):
        """Test the SEQUE-NCD example: Fibonacci sequence."""
        seq = parse_sequence("55, ?, 144, 233, 377, 610")
        assert seq.values == [55, None, 144, 233, 377, 610]
        assert seq.missing_indices == [1]
        assert seq.known_values == [55, 144, 233, 377, 610]

    def test_various_missing_indicators(self):
        """Test different ways to indicate missing values."""
        test_cases = [
            ("1, ?, 3", [1, None, 3]),
            ("1, x, 3", [1, None, 3]),
            ("1, X, 3", [1, None, 3]),
            ("1, _, 3", [1, None, 3]),
            ("1, -, 3", [1, None, 3]),
        ]
        for input_str, expected in test_cases:
            seq = parse_sequence(input_str)
            assert seq.values == expected, f"Failed for input: {input_str}"

    def test_with_brackets(self):
        """Test parsing with brackets."""
        seq = parse_sequence("[1, 2, ?, 4]")
        assert seq.values == [1, 2, None, 4]

    def test_floats(self):
        """Test parsing floating point numbers."""
        seq = parse_sequence("1.5, 2.5, ?, 4.5")
        assert seq.values == [1.5, 2.5, None, 4.5]

    def test_negative_numbers(self):
        """Test parsing negative numbers."""
        seq = parse_sequence("-5, -3, ?, 1, 3")
        assert seq.values == [-5, -3, None, 1, 3]

    def test_no_spaces(self):
        """Test parsing without spaces."""
        seq = parse_sequence("1,2,?,4,5")
        assert seq.values == [1, 2, None, 4, 5]

    def test_extra_spaces(self):
        """Test parsing with extra whitespace."""
        seq = parse_sequence("  1 ,  2  , ? ,  4  ")
        assert seq.values == [1, 2, None, 4]

    def test_empty_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError):
            parse_sequence("")

    def test_all_missing_raises(self):
        """Test that all-missing sequence raises ValueError."""
        with pytest.raises(ValueError):
            parse_sequence("?, ?, ?")


class TestFormatSequence:
    """Tests for format_sequence function."""

    def test_format_simple(self):
        """Test formatting a simple sequence."""
        seq = parse_sequence("1, 2, 3")
        formatted = format_sequence(seq)
        assert formatted == "1, 2, 3"

    def test_format_with_missing(self):
        """Test formatting with missing values highlighted."""
        seq = parse_sequence("1, ?, 3")
        formatted = format_sequence(seq, highlight_missing=True)
        assert formatted == "1, [?], 3"

    def test_format_without_highlight(self):
        """Test formatting without highlight."""
        seq = parse_sequence("1, ?, 3")
        formatted = format_sequence(seq, highlight_missing=False)
        assert formatted == "1, ?, 3"


class TestSequenceProperties:
    """Tests for Sequence class properties."""

    def test_known_values(self):
        """Test known_values property."""
        seq = parse_sequence("1, ?, 3, ?, 5")
        assert seq.known_values == [1, 3, 5]

    def test_known_pairs(self):
        """Test known_pairs property."""
        seq = parse_sequence("1, ?, 3, ?, 5")
        assert seq.known_pairs == [(0, 1), (2, 3), (4, 5)]

    def test_len(self):
        """Test __len__ method."""
        seq = parse_sequence("1, 2, 3, 4, 5")
        assert len(seq) == 5
