"""Tests for ensemble combiner."""

import pytest
from seq.parser import parse_sequence
from seq.ensemble import EnsembleDetector, analyze_sequence


class TestEnsembleDetector:
    """Tests for the ensemble detector."""

    @pytest.mark.asyncio
    async def test_fibonacci_ensemble(self):
        """Test ensemble detection of Fibonacci sequence."""
        result = await analyze_sequence(
            "55, ?, 144, 233, 377, 610",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(1)
        assert best is not None
        assert best.value == 89
        assert best.combined_confidence >= 0.9

    @pytest.mark.asyncio
    async def test_arithmetic_ensemble(self):
        """Test ensemble detection of arithmetic sequence."""
        result = await analyze_sequence(
            "2, 4, ?, 8, 10",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(2)
        assert best is not None
        assert best.value == 6

    @pytest.mark.asyncio
    async def test_geometric_ensemble(self):
        """Test ensemble detection of geometric sequence."""
        result = await analyze_sequence(
            "2, 4, ?, 16, 32",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(2)
        assert best is not None
        assert best.value == 8

    @pytest.mark.asyncio
    async def test_squares_ensemble(self):
        """Test ensemble detection of square numbers."""
        result = await analyze_sequence(
            "1, 4, 9, ?, 25, 36",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(3)
        assert best is not None
        assert best.value == 16

    @pytest.mark.asyncio
    async def test_cubes_ensemble(self):
        """Test ensemble detection of cube numbers."""
        result = await analyze_sequence(
            "1, 8, 27, ?, 125",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(3)
        assert best is not None
        assert best.value == 64

    @pytest.mark.asyncio
    async def test_multiple_missing(self):
        """Test ensemble with multiple missing values."""
        result = await analyze_sequence(
            "1, ?, 3, ?, 5",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best_1 = result.get_best(1)
        best_3 = result.get_best(3)

        assert best_1 is not None and best_1.value == 2
        assert best_3 is not None and best_3.value == 4

    @pytest.mark.asyncio
    async def test_get_top_k(self):
        """Test getting top k predictions."""
        result = await analyze_sequence(
            "2, 4, ?, 16, 32",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        top_3 = result.get_top_k(2, k=3)
        assert len(top_3) >= 1
        # Best prediction should be 8
        assert top_3[0].value == 8


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction class."""

    @pytest.mark.asyncio
    async def test_methods_property(self):
        """Test that methods property returns contributing methods."""
        result = await analyze_sequence(
            "55, ?, 144, 233, 377, 610",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(1)
        assert best is not None
        assert "rule-based" in best.methods

    @pytest.mark.asyncio
    async def test_best_explanation(self):
        """Test that best_explanation returns a meaningful explanation."""
        result = await analyze_sequence(
            "55, ?, 144, 233, 377, 610",
            use_rules=True,
            use_oeis=False,
            use_ml=False,
        )

        best = result.get_best(1)
        assert best is not None
        assert best.best_explanation is not None
        assert "Fibonacci" in best.best_explanation
