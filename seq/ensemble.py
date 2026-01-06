"""Ensemble system for combining predictions from multiple detectors."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .detectors.base import Detector, DetectionResult, Prediction, Sequence
from .detectors.rules import RuleBasedDetector
from .detectors.oeis import OEISDetector
from .detectors.ml import MLDetector
from .detectors.constants import ConstantsDetector


@dataclass
class EnsemblePrediction:
    """A combined prediction from multiple methods."""

    value: int | float
    combined_confidence: float
    predictions: list[Prediction]  # Individual predictions that contributed

    @property
    def methods(self) -> list[str]:
        """Get list of methods that contributed to this prediction."""
        return list(set(p.method for p in self.predictions))

    @property
    def best_explanation(self) -> Optional[str]:
        """Get the most informative explanation."""
        # Prefer rule-based explanations, then OEIS, then ML
        for method_prefix in ["rule", "oeis", "ml"]:
            for pred in sorted(self.predictions, key=lambda p: -p.confidence):
                if pred.method.startswith(method_prefix) and pred.explanation:
                    return pred.explanation
        return None


@dataclass
class EnsembleResult:
    """Complete result from ensemble prediction."""

    predictions: dict[int, list[EnsemblePrediction]] = field(default_factory=dict)

    def get_best(self, index: int) -> Optional[EnsemblePrediction]:
        """Get the highest confidence prediction for an index."""
        if index not in self.predictions or not self.predictions[index]:
            return None
        return max(self.predictions[index], key=lambda p: p.combined_confidence)

    def get_top_k(self, index: int, k: int = 5) -> list[EnsemblePrediction]:
        """Get top k predictions for an index."""
        if index not in self.predictions:
            return []
        sorted_preds = sorted(
            self.predictions[index],
            key=lambda p: -p.combined_confidence
        )
        return sorted_preds[:k]


class EnsembleDetector:
    """
    Combines multiple detection methods into a unified prediction system.

    Aggregates predictions from rule-based, OEIS, and ML detectors,
    then combines them using weighted voting based on confidence and
    method reliability.
    """

    def __init__(
        self,
        use_rules: bool = True,
        use_oeis: bool = True,
        use_ml: bool = True,
        use_constants: bool = True,
        method_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the ensemble detector.

        Args:
            use_rules: Whether to use rule-based detection
            use_oeis: Whether to use OEIS lookup
            use_ml: Whether to use ML models
            use_constants: Whether to use mathematical constants detection
            method_weights: Custom weights for each method
        """
        self.detectors: list[Detector] = []

        if use_rules:
            self.detectors.append(RuleBasedDetector())
        if use_oeis:
            self.detectors.append(OEISDetector())
        if use_ml:
            self.detectors.append(MLDetector())
        if use_constants:
            self.detectors.append(ConstantsDetector())

        # Default weights based on expected reliability
        self.method_weights = method_weights or {
            "rule-based": 1.0,  # High weight for mathematical patterns
            "oeis": 0.9,        # High weight for known sequences
            "constants": 0.95,  # High weight for known constants (very reliable when matched)
            "ml/transformer": 0.6,  # Medium weight for ML
            "ml/lstm": 0.6,
        }

    async def detect(self, sequence: Sequence) -> EnsembleResult:
        """
        Run all detectors and combine results.

        Args:
            sequence: The sequence to analyze

        Returns:
            Combined predictions with confidence scores
        """
        # Run all detectors in parallel
        tasks = [detector.detect(sequence) for detector in self.detectors]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all predictions
        all_predictions: dict[int, list[Prediction]] = {}

        for result in results:
            if isinstance(result, Exception):
                continue

            for idx, preds in result.predictions.items():
                if idx not in all_predictions:
                    all_predictions[idx] = []
                all_predictions[idx].extend(preds)

        # Combine predictions for each index
        ensemble_result = EnsembleResult()

        for idx, preds in all_predictions.items():
            combined = self._combine_predictions(preds)
            ensemble_result.predictions[idx] = combined

        return ensemble_result

    def _combine_predictions(
        self, predictions: list[Prediction]
    ) -> list[EnsemblePrediction]:
        """
        Combine predictions for a single position.

        Groups predictions by value and computes combined confidence.

        Args:
            predictions: List of predictions from different methods

        Returns:
            List of combined predictions
        """
        if not predictions:
            return []

        # Group predictions by value (with tolerance for floats)
        value_groups: dict[int | float, list[Prediction]] = {}

        for pred in predictions:
            # Round to handle float comparison - prefer integer if close
            if isinstance(pred.value, float) and abs(pred.value - round(pred.value)) < 0.1:
                key = int(round(pred.value))
            elif isinstance(pred.value, float):
                key = round(pred.value, 3)
            else:
                key = pred.value

            # Find if there's a close match
            matched_key = None
            for existing_key in value_groups:
                if abs(existing_key - key) < 0.5:  # Wider tolerance for grouping
                    matched_key = existing_key
                    break

            if matched_key is not None:
                # Prefer integer keys over float keys
                if isinstance(key, int) and isinstance(matched_key, float):
                    value_groups[key] = value_groups.pop(matched_key)
                    value_groups[key].append(pred)
                else:
                    value_groups[matched_key].append(pred)
            else:
                value_groups[key] = [pred]

        # Compute combined confidence for each value
        combined_predictions = []

        for value, preds in value_groups.items():
            # Weighted combination
            weighted_sum = 0.0
            weight_total = 0.0

            for pred in preds:
                weight = self._get_method_weight(pred.method)
                weighted_sum += pred.confidence * weight
                weight_total += weight

            combined_confidence = weighted_sum / weight_total if weight_total > 0 else 0.0

            # Boost confidence if multiple methods agree
            agreement_bonus = min(0.2, 0.1 * (len(preds) - 1))
            combined_confidence = min(1.0, combined_confidence + agreement_bonus)

            # Use integer value if it's a whole number
            final_value = int(value) if isinstance(value, float) and value == int(value) else value

            combined_predictions.append(EnsemblePrediction(
                value=final_value,
                combined_confidence=combined_confidence,
                predictions=preds,
            ))

        # Sort by confidence
        combined_predictions.sort(key=lambda p: -p.combined_confidence)

        return combined_predictions

    def _get_method_weight(self, method: str) -> float:
        """Get weight for a detection method."""
        # Try exact match first
        if method in self.method_weights:
            return self.method_weights[method]

        # Try prefix match
        for key, weight in self.method_weights.items():
            if method.startswith(key):
                return weight

        return 0.5  # Default weight


async def analyze_sequence(
    sequence_str: str,
    use_rules: bool = True,
    use_oeis: bool = True,
    use_ml: bool = True,
) -> EnsembleResult:
    """
    Convenience function to analyze a sequence string.

    Args:
        sequence_str: String representation of sequence (e.g., "1, 2, ?, 4, 5")
        use_rules: Whether to use rule-based detection
        use_oeis: Whether to use OEIS lookup
        use_ml: Whether to use ML models

    Returns:
        EnsembleResult with predictions
    """
    from .parser import parse_sequence

    sequence = parse_sequence(sequence_str)
    detector = EnsembleDetector(
        use_rules=use_rules,
        use_oeis=use_oeis,
        use_ml=use_ml,
    )

    return await detector.detect(sequence)
