"""Tests for rule-based sequence detection."""

import pytest
from seq.parser import parse_sequence
from seq.detectors.rules import RuleBasedDetector


@pytest.fixture
def detector():
    """Create a rule-based detector."""
    return RuleBasedDetector()


class TestArithmeticSequence:
    """Tests for arithmetic sequence detection."""

    @pytest.mark.asyncio
    async def test_simple_arithmetic(self, detector):
        """Test detection of simple arithmetic sequence."""
        seq = parse_sequence("2, 4, ?, 8, 10")
        result = await detector.detect(seq)

        best = result.get_best(2)
        assert best is not None
        assert best.value == 6
        assert best.confidence > 0.9
        assert best.pattern_name == "Arithmetic"

    @pytest.mark.asyncio
    async def test_arithmetic_negative_diff(self, detector):
        """Test arithmetic sequence with negative difference."""
        seq = parse_sequence("10, 8, ?, 4, 2")
        result = await detector.detect(seq)

        best = result.get_best(2)
        assert best is not None
        assert best.value == 6

    @pytest.mark.asyncio
    async def test_arithmetic_first_position(self, detector):
        """Test predicting first position in arithmetic sequence."""
        seq = parse_sequence("?, 4, 6, 8, 10")
        result = await detector.detect(seq)

        best = result.get_best(0)
        assert best is not None
        assert best.value == 2


class TestGeometricSequence:
    """Tests for geometric sequence detection."""

    @pytest.mark.asyncio
    async def test_simple_geometric(self, detector):
        """Test detection of simple geometric sequence."""
        seq = parse_sequence("2, 4, ?, 16, 32")
        result = await detector.detect(seq)

        best = result.get_best(2)
        assert best is not None
        assert best.value == 8
        assert best.pattern_name == "Geometric"

    @pytest.mark.asyncio
    async def test_powers_of_two(self, detector):
        """Test powers of 2 sequence."""
        seq = parse_sequence("1, 2, 4, ?, 16, 32")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 8


class TestFibonacciSequence:
    """Tests for Fibonacci sequence detection."""

    @pytest.mark.asyncio
    async def test_fibonacci_standard(self, detector):
        """Test standard Fibonacci sequence."""
        seq = parse_sequence("1, 1, 2, 3, ?, 8, 13")
        result = await detector.detect(seq)

        best = result.get_best(4)
        assert best is not None
        assert best.value == 5
        assert "Fibonacci" in best.pattern_name

    @pytest.mark.asyncio
    async def test_fibonacci_seque_ncd_example(self, detector):
        """Test the SEQUE-NCD puzzle example."""
        seq = parse_sequence("55, ?, 144, 233, 377, 610")
        result = await detector.detect(seq)

        best = result.get_best(1)
        assert best is not None
        assert best.value == 89
        assert best.confidence > 0.8

    @pytest.mark.asyncio
    async def test_lucas_numbers(self, detector):
        """Test Lucas numbers (Fibonacci-like with different seeds)."""
        seq = parse_sequence("2, 1, 3, 4, ?, 11, 18")
        result = await detector.detect(seq)

        best = result.get_best(4)
        assert best is not None
        assert best.value == 7


class TestPolynomialSequence:
    """Tests for polynomial sequence detection."""

    @pytest.mark.asyncio
    async def test_squares(self, detector):
        """Test square numbers."""
        seq = parse_sequence("1, 4, 9, ?, 25, 36")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 16

    @pytest.mark.asyncio
    async def test_cubes(self, detector):
        """Test cube numbers."""
        seq = parse_sequence("1, 8, 27, ?, 125")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 64


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_too_few_known_values(self, detector):
        """Test with minimal known values."""
        seq = parse_sequence("1, ?")
        result = await detector.detect(seq)
        # Should still work but with lower confidence
        assert result is not None

    @pytest.mark.asyncio
    async def test_constant_sequence(self, detector):
        """Test constant sequence (difference = 0)."""
        seq = parse_sequence("5, 5, ?, 5, 5")
        result = await detector.detect(seq)

        best = result.get_best(2)
        assert best is not None
        assert best.value == 5

    @pytest.mark.asyncio
    async def test_multiple_missing(self, detector):
        """Test sequence with multiple missing values."""
        seq = parse_sequence("2, ?, 6, ?, 10")
        result = await detector.detect(seq)

        best_2 = result.get_best(1)
        best_4 = result.get_best(3)

        assert best_2 is not None and best_2.value == 4
        assert best_4 is not None and best_4.value == 8
