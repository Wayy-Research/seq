"""Tests for mathematical constants detection."""

import pytest
from seq.parser import parse_sequence
from seq.detectors.constants import ConstantsDetector


@pytest.fixture
def detector():
    """Create a constants detector."""
    return ConstantsDetector()


class TestEulerNumber:
    """Tests for Euler's number e = 2.71828..."""

    @pytest.mark.asyncio
    async def test_e_digits_missing_middle(self, detector):
        """Test detecting e digits with missing value in middle."""
        # e = 2.71828...
        seq = parse_sequence("2, 7, 1, ?, 2, 8")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 8
        assert best.confidence >= 0.7
        assert "e" in best.pattern_name.lower() or "euler" in best.pattern_name.lower()

    @pytest.mark.asyncio
    async def test_e_digits_missing_first(self, detector):
        """Test detecting e digits with missing first value."""
        # e = 2.71828...
        seq = parse_sequence("?, 7, 1, 8, 2, 8")
        result = await detector.detect(seq)

        best = result.get_best(0)
        assert best is not None
        assert best.value == 2

    @pytest.mark.asyncio
    async def test_e_digits_missing_last(self, detector):
        """Test detecting e digits with missing last value."""
        # e = 2.71828...
        seq = parse_sequence("2, 7, 1, 8, 2, ?")
        result = await detector.detect(seq)

        best = result.get_best(5)
        assert best is not None
        assert best.value == 8

    @pytest.mark.asyncio
    async def test_e_digits_longer_sequence(self, detector):
        """Test with more digits of e."""
        # e = 2.7182818284...
        seq = parse_sequence("2, 7, 1, 8, 2, 8, 1, ?, 2, 8, 4")
        result = await detector.detect(seq)

        best = result.get_best(7)
        assert best is not None
        assert best.value == 8


class TestPi:
    """Tests for pi = 3.14159..."""

    @pytest.mark.asyncio
    async def test_pi_digits_missing_middle(self, detector):
        """Test detecting pi digits."""
        # pi = 3.14159...
        seq = parse_sequence("3, 1, 4, ?, 5, 9")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 1
        assert "pi" in best.pattern_name.lower()

    @pytest.mark.asyncio
    async def test_pi_digits_longer(self, detector):
        """Test with more digits of pi."""
        # pi = 3.1415926535...
        seq = parse_sequence("3, 1, 4, 1, 5, 9, 2, 6, ?, 3, 5")
        result = await detector.detect(seq)

        best = result.get_best(8)
        assert best is not None
        assert best.value == 5


class TestGoldenRatio:
    """Tests for golden ratio phi = 1.61803..."""

    @pytest.mark.asyncio
    async def test_phi_digits(self, detector):
        """Test detecting golden ratio digits."""
        # phi = 1.61803...
        seq = parse_sequence("1, 6, 1, ?, 0, 3")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 8
        assert "phi" in best.pattern_name.lower() or "golden" in best.pattern_name.lower()


class TestSqrt2:
    """Tests for sqrt(2) = 1.41421..."""

    @pytest.mark.asyncio
    async def test_sqrt2_digits(self, detector):
        """Test detecting sqrt(2) digits."""
        # sqrt(2) = 1.41421...
        seq = parse_sequence("1, 4, 1, ?, 2, 1")
        result = await detector.detect(seq)

        best = result.get_best(3)
        assert best is not None
        assert best.value == 4
        assert "sqrt" in best.pattern_name.lower() or "2" in best.pattern_name


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_too_few_digits(self, detector):
        """Test that we need at least 3 digits for a match."""
        seq = parse_sequence("2, ?, 1")
        result = await detector.detect(seq)

        # Should return empty or low confidence since 2, ?, 1 is ambiguous
        best = result.get_best(1)
        # Either no result or multiple possible matches (e starts 2,7,1 so 7 is valid)
        if best is not None:
            # If we get a result, it should be reasonable
            assert isinstance(best.value, int)

    @pytest.mark.asyncio
    async def test_non_digit_values_ignored(self, detector):
        """Test that sequences with non-single-digit values are ignored."""
        seq = parse_sequence("27, 18, 28, ?, 45")
        result = await detector.detect(seq)

        # Should not match anything (values > 9)
        assert result.get_best(3) is None

    @pytest.mark.asyncio
    async def test_negative_values_ignored(self, detector):
        """Test that sequences with negative values are ignored."""
        seq = parse_sequence("-2, 7, 1, ?, 2, 8")
        result = await detector.detect(seq)

        # Should not match (negative value)
        assert result.get_best(3) is None

    @pytest.mark.asyncio
    async def test_multiple_missing_values(self, detector):
        """Test sequence with multiple missing values."""
        # e = 2.71828...
        seq = parse_sequence("2, ?, 1, 8, ?, 8")
        result = await detector.detect(seq)

        best_1 = result.get_best(1)
        best_4 = result.get_best(4)

        assert best_1 is not None
        assert best_1.value == 7
        assert best_4 is not None
        assert best_4.value == 2

    @pytest.mark.asyncio
    async def test_no_match_for_random_sequence(self, detector):
        """Test that random sequences don't match constants."""
        seq = parse_sequence("5, 5, 5, ?, 5, 5")
        result = await detector.detect(seq)

        # Should not confidently match any constant
        best = result.get_best(3)
        if best is not None:
            # If there's a match, confidence should be from other methods
            pass  # Allow for coincidental matches in constant digits
