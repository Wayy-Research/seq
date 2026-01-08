"""Rule-based sequence detectors for common mathematical patterns."""

import numpy as np
from typing import Optional
from .base import Detector, DetectionResult, Prediction, Sequence


class RuleBasedDetector(Detector):
    """
    Detector that uses mathematical rules to identify common sequence patterns.

    Supports:
    - Arithmetic sequences (constant difference)
    - Geometric sequences (constant ratio)
    - Fibonacci-like sequences (each term = sum of previous two)
    - Polynomial sequences (differences stabilize)
    - Linear recurrence relations
    - Alternating ratio sequences (e.g., ×5, ×2, ×5, ×2...)
    """

    name = "rule-based"

    async def detect(self, sequence: Sequence) -> DetectionResult:
        """Detect patterns and predict missing values."""
        result = DetectionResult()

        # Try each pattern detector
        detectors = [
            self._detect_arithmetic,
            self._detect_geometric,
            self._detect_alternating_ratio,
            self._detect_fibonacci,
            self._detect_polynomial,
            self._detect_linear_recurrence,
        ]

        for detector in detectors:
            try:
                predictions = detector(sequence)
                for idx, pred in predictions.items():
                    result.add_prediction(idx, pred)
            except Exception:
                # Pattern doesn't fit, continue to next
                continue

        return result

    def _detect_arithmetic(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect arithmetic sequences (constant difference).

        Example: 2, 4, 6, 8 (difference = 2)
        """
        known = sequence.known_pairs
        if len(known) < 2:
            return {}

        # Calculate differences between consecutive known values
        differences = []
        for i in range(len(known) - 1):
            idx1, val1 = known[i]
            idx2, val2 = known[i + 1]
            # Normalize by index gap
            gap = idx2 - idx1
            diff = (val2 - val1) / gap
            differences.append(diff)

        # Check if differences are constant
        if not differences:
            return {}

        avg_diff = np.mean(differences)
        std_diff = np.std(differences) if len(differences) > 1 else 0

        # Confidence based on consistency of differences
        if avg_diff == 0:
            confidence = 1.0 if std_diff == 0 else 0.0
        else:
            confidence = max(0.0, 1.0 - (std_diff / abs(avg_diff))) if std_diff < abs(avg_diff) else 0.0

        if confidence < 0.5:
            return {}

        # Predict missing values
        predictions = {}
        first_idx, first_val = known[0]

        for missing_idx in sequence.missing_indices:
            predicted_val = first_val + avg_diff * (missing_idx - first_idx)
            # Round to int if close
            if abs(predicted_val - round(predicted_val)) < 0.001:
                predicted_val = int(round(predicted_val))

            predictions[missing_idx] = Prediction(
                value=predicted_val,
                confidence=confidence,
                method=self.name,
                pattern_name="Arithmetic",
                explanation=f"Arithmetic sequence with common difference {avg_diff:.2f}"
            )

        return predictions

    def _detect_geometric(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect geometric sequences (constant ratio).

        Example: 2, 4, 8, 16 (ratio = 2)
        """
        known = sequence.known_pairs
        if len(known) < 2:
            return {}

        # Check for zeros (geometric won't work)
        if any(v == 0 for _, v in known):
            return {}

        # Calculate ratios between consecutive known values
        ratios = []
        for i in range(len(known) - 1):
            idx1, val1 = known[i]
            idx2, val2 = known[i + 1]
            gap = idx2 - idx1
            # ratio^gap = val2/val1, so ratio = (val2/val1)^(1/gap)
            try:
                ratio = (val2 / val1) ** (1 / gap)
                ratios.append(ratio)
            except (ZeroDivisionError, ValueError):
                return {}

        if not ratios:
            return {}

        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios) if len(ratios) > 1 else 0

        # Confidence based on consistency of ratios
        if avg_ratio == 0:
            return {}

        confidence = max(0.0, 1.0 - (std_ratio / abs(avg_ratio))) if std_ratio < abs(avg_ratio) else 0.0

        if confidence < 0.5:
            return {}

        # Predict missing values
        predictions = {}
        first_idx, first_val = known[0]

        for missing_idx in sequence.missing_indices:
            predicted_val = first_val * (avg_ratio ** (missing_idx - first_idx))
            # Round to int if close
            if abs(predicted_val - round(predicted_val)) < 0.001:
                predicted_val = int(round(predicted_val))

            predictions[missing_idx] = Prediction(
                value=predicted_val,
                confidence=confidence,
                method=self.name,
                pattern_name="Geometric",
                explanation=f"Geometric sequence with common ratio {avg_ratio:.2f}"
            )

        return predictions

    def _detect_alternating_ratio(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect alternating ratio sequences (two ratios that alternate).

        Example: 1, 5, 10, 50, 100, 500, 1000 (ratios: ×5, ×2, ×5, ×2, ×5, ×2)
        """
        known = sequence.known_pairs
        if len(known) < 4:  # Need at least 4 values to detect alternating pattern
            return {}

        # Check for zeros (ratios won't work)
        if any(v == 0 for _, v in known):
            return {}

        # Build index -> value map
        val_map = {idx: val for idx, val in known}
        indices = sorted(val_map.keys())

        # Calculate ratios for consecutive indices
        ratios_with_positions = []
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            # Only consider consecutive indices (gap of 1)
            if idx2 - idx1 == 1:
                try:
                    ratio = val_map[idx2] / val_map[idx1]
                    ratios_with_positions.append((idx1, ratio))
                except (ZeroDivisionError, ValueError):
                    return {}

        if len(ratios_with_positions) < 3:  # Need at least 3 ratios to detect alternation
            return {}

        # Separate ratios by position parity (even/odd index)
        even_ratios = [r for idx, r in ratios_with_positions if idx % 2 == 0]
        odd_ratios = [r for idx, r in ratios_with_positions if idx % 2 == 1]

        if not even_ratios or not odd_ratios:
            return {}

        # Check if each group is consistent
        avg_even = np.mean(even_ratios)
        avg_odd = np.mean(odd_ratios)
        std_even = np.std(even_ratios) if len(even_ratios) > 1 else 0
        std_odd = np.std(odd_ratios) if len(odd_ratios) > 1 else 0

        # Both groups must be internally consistent
        if avg_even == 0 or avg_odd == 0:
            return {}

        even_consistent = std_even < 0.01 * abs(avg_even) or std_even < 0.001
        odd_consistent = std_odd < 0.01 * abs(avg_odd) or std_odd < 0.001

        if not (even_consistent and odd_consistent):
            return {}

        # The two ratios must be different (otherwise it's just geometric)
        if abs(avg_even - avg_odd) < 0.01 * max(abs(avg_even), abs(avg_odd)):
            return {}

        # High confidence if pattern holds
        confidence = 0.95

        # Predict missing values
        predictions = {}
        for missing_idx in sequence.missing_indices:
            # Find the closest known value to work from
            predicted_val = None

            # Try to compute forward from a known predecessor
            if (missing_idx - 1) in val_map:
                prev_idx = missing_idx - 1
                ratio = avg_even if prev_idx % 2 == 0 else avg_odd
                predicted_val = val_map[prev_idx] * ratio

            # Try to compute backward from a known successor
            elif (missing_idx + 1) in val_map:
                next_idx = missing_idx + 1
                # The ratio from missing_idx to next_idx
                ratio = avg_even if missing_idx % 2 == 0 else avg_odd
                predicted_val = val_map[next_idx] / ratio

            if predicted_val is not None:
                # Round to int if close
                if abs(predicted_val - round(predicted_val)) < 0.001:
                    predicted_val = int(round(predicted_val))

                # Format ratios for explanation
                r1 = avg_even if abs(avg_even - round(avg_even)) > 0.01 else int(round(avg_even))
                r2 = avg_odd if abs(avg_odd - round(avg_odd)) > 0.01 else int(round(avg_odd))

                predictions[missing_idx] = Prediction(
                    value=predicted_val,
                    confidence=confidence,
                    method=self.name,
                    pattern_name="Alternating Ratio",
                    explanation=f"Alternating ratios: ×{r1}, ×{r2}, ×{r1}, ×{r2}..."
                )

        return predictions

    def _detect_fibonacci(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect Fibonacci-like sequences (a[n] = a[n-1] + a[n-2]).

        Example: 1, 1, 2, 3, 5, 8, 13
        """
        known = sequence.known_pairs
        if len(known) < 3:
            return {}

        # Check if the pattern a[n] = a[n-1] + a[n-2] holds for known values
        matches = 0
        total_checks = 0

        # Build index -> value map from known values
        val_map = {idx: val for idx, val in known}

        for idx, val in known:
            if (idx - 1) in val_map and (idx - 2) in val_map:
                expected = val_map[idx - 1] + val_map[idx - 2]
                if abs(expected - val) < 0.001:
                    matches += 1
                total_checks += 1

        if total_checks == 0:
            return {}

        confidence = matches / total_checks

        if confidence < 0.8:  # Require high confidence for Fibonacci
            return {}

        # Predict missing values
        predictions = {}

        # For each missing index, try to compute from neighbors
        for missing_idx in sequence.missing_indices:
            predicted_val = self._predict_fibonacci_value(missing_idx, val_map, sequence)
            if predicted_val is not None:
                predictions[missing_idx] = Prediction(
                    value=predicted_val,
                    confidence=confidence,
                    method=self.name,
                    pattern_name="Fibonacci",
                    explanation="Fibonacci-like sequence: a[n] = a[n-1] + a[n-2]"
                )

        return predictions

    def _predict_fibonacci_value(
        self, idx: int, val_map: dict[int, float], sequence: Sequence
    ) -> Optional[int | float]:
        """Predict a Fibonacci value given known values."""
        # Try forward: if we know idx-1 and idx-2, compute idx
        if (idx - 1) in val_map and (idx - 2) in val_map:
            val = val_map[idx - 1] + val_map[idx - 2]
            if abs(val - round(val)) < 0.1:
                return int(round(val))
            return val

        # Try backward: if we know idx+1 and idx+2, compute idx
        # a[idx] = a[idx+2] - a[idx+1]
        if (idx + 1) in val_map and (idx + 2) in val_map:
            val = val_map[idx + 2] - val_map[idx + 1]
            if abs(val - round(val)) < 0.1:
                return int(round(val))
            return val

        # Try another backward: if we know idx+1 and idx-1
        # a[idx] = a[idx+1] - a[idx-1]
        if (idx + 1) in val_map and (idx - 1) in val_map:
            val = val_map[idx + 1] - val_map[idx - 1]
            if abs(val - round(val)) < 0.1:
                return int(round(val))
            return val

        return None

    def _detect_polynomial(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect polynomial sequences by fitting polynomials of various degrees.

        For a polynomial of degree n, the n-th differences are constant.
        - Degree 2 (quadratic): squares, triangular numbers, etc.
        - Degree 3 (cubic): cubes, etc.

        Works with gaps in the sequence by using polynomial fitting directly.

        IMPORTANT: A degree-n polynomial can fit ANY n+1 points perfectly (overfitting).
        We require significantly more points than the degree to ensure the pattern is real.
        """
        known = sequence.known_pairs
        if len(known) < 4:  # Need at least 4 points for meaningful polynomial fit
            return {}

        x_known = np.array([idx for idx, _ in known])
        y_known = np.array([val for _, val in known])

        # Try polynomial degrees 2, 3, 4 (degree 1 is arithmetic, handled separately)
        best_result = None
        best_confidence = 0.0

        for degree in [2, 3, 4]:
            # CRITICAL: Require at least 2*degree points to prevent overfitting
            # A degree-n polynomial fits ANY n+1 points perfectly, so we need
            # significantly more points to verify the pattern is real
            min_points_required = max(degree + 2, 2 * degree)
            if len(known) < min_points_required:
                continue

            try:
                coeffs = np.polyfit(x_known, y_known, degree)
                poly = np.poly1d(coeffs)

                # Verify fit - check if polynomial exactly matches known values
                y_pred = poly(x_known)
                max_error = np.max(np.abs(y_pred - y_known))

                # For integer sequences, we expect near-perfect fit
                if max_error > 0.5:
                    continue

                # Calculate confidence based on fit quality
                # Perfect fit = 1.0, any error reduces confidence
                confidence = max(0.0, 1.0 - max_error)

                # Check if all predictions would be integers (common for polynomial sequences)
                all_integer = True
                for missing_idx in sequence.missing_indices:
                    pred_val = poly(missing_idx)
                    if abs(pred_val - round(pred_val)) > 0.1:
                        all_integer = False
                        break

                # Boost confidence for integer results
                if all_integer:
                    confidence = min(1.0, confidence + 0.1)

                # Moderate penalty for higher degree polynomials (Occam's razor)
                # Higher degrees are more likely to overfit
                confidence -= 0.08 * (degree - 2)

                # Small penalty when we're at the exact minimum points
                # Having more points = more confidence the pattern is real
                excess_points = len(known) - min_points_required
                if excess_points == 0:
                    confidence -= 0.05

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = (poly, degree, confidence)

            except Exception:
                continue

        if best_result is None or best_confidence < 0.8:
            return {}

        poly, degree, confidence = best_result

        # Generate predictions
        predictions = {}
        for missing_idx in sequence.missing_indices:
            predicted_val = poly(missing_idx)
            if abs(predicted_val - round(predicted_val)) < 0.1:
                predicted_val = int(round(predicted_val))

            # Determine pattern name based on sequence type
            pattern_name = f"Polynomial (degree {degree})"
            explanation = f"Polynomial sequence of degree {degree}"

            # Try to identify common sequences
            if degree == 2:
                # Check if it's squares: n^2
                test_vals = [poly(i) for i in range(10)]
                if all(abs(test_vals[i] - (i+1)**2) < 0.1 for i in range(min(5, len(test_vals))) if i < len(test_vals)):
                    pattern_name = "Square numbers"
                    explanation = "Square numbers: n²"
                # Check if it's triangular: n(n+1)/2
                elif all(abs(test_vals[i] - (i+1)*(i+2)//2) < 0.1 for i in range(min(5, len(test_vals))) if i < len(test_vals)):
                    pattern_name = "Triangular numbers"
                    explanation = "Triangular numbers: n(n+1)/2"
            elif degree == 3:
                # Check if it's cubes: n^3
                test_vals = [poly(i) for i in range(10)]
                if all(abs(test_vals[i] - (i+1)**3) < 0.1 for i in range(min(5, len(test_vals))) if i < len(test_vals)):
                    pattern_name = "Cube numbers"
                    explanation = "Cube numbers: n³"

            predictions[missing_idx] = Prediction(
                value=predicted_val,
                confidence=confidence,
                method=self.name,
                pattern_name=pattern_name,
                explanation=explanation
            )

        return predictions

    def _detect_linear_recurrence(self, sequence: Sequence) -> dict[int, Prediction]:
        """
        Detect general linear recurrence relations.

        a[n] = c1*a[n-1] + c2*a[n-2] + ... + ck*a[n-k]

        This generalizes Fibonacci (c1=1, c2=1) and other patterns.
        """
        known = sequence.known_pairs
        if len(known) < 5:
            return {}

        # Try recurrence orders 2 and 3, pick the best fit
        best_predictions = {}
        best_confidence = 0.0

        for order in [2, 3]:
            predictions = self._try_recurrence_order(sequence, known, order)
            if predictions:
                # Get average confidence from predictions
                avg_conf = sum(p.confidence for p in predictions.values()) / len(predictions)
                if avg_conf > best_confidence:
                    best_confidence = avg_conf
                    best_predictions = predictions

        return best_predictions

    def _try_recurrence_order(
        self, sequence: Sequence, known: list[tuple[int, float]], order: int
    ) -> dict[int, Prediction]:
        """Try to fit a recurrence relation of given order."""
        # Build index -> value map
        val_map = {idx: val for idx, val in known}

        # Find consecutive runs long enough to fit recurrence
        indices = sorted(val_map.keys())
        runs = []
        current_run = [indices[0]]

        for i in range(1, len(indices)):
            if indices[i] == indices[i - 1] + 1:
                current_run.append(indices[i])
            else:
                if len(current_run) > order:
                    runs.append(current_run)
                current_run = [indices[i]]
        if len(current_run) > order:
            runs.append(current_run)

        if not runs:
            return {}

        # Use the longest run to fit recurrence
        longest_run = max(runs, key=len)
        if len(longest_run) < order + 2:
            return {}

        # Build system of equations
        values = [val_map[idx] for idx in longest_run]
        n_equations = len(values) - order

        A = np.zeros((n_equations, order))
        b = np.zeros(n_equations)

        for i in range(n_equations):
            for j in range(order):
                A[i, j] = values[i + order - 1 - j]
            b[i] = values[i + order]

        try:
            # Solve least squares
            coeffs, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)

            # Verify fit
            predicted = A @ coeffs
            error = np.mean(np.abs(predicted - b))
            max_val = np.max(np.abs(b)) + 0.001
            confidence = max(0.0, 1.0 - (error / max_val))

            if confidence < 0.95:  # Require very high confidence
                return {}

            # Generate predictions
            predictions = {}
            for missing_idx in sequence.missing_indices:
                # Try to compute from known values
                can_compute = all(
                    (missing_idx - j - 1) in val_map
                    for j in range(order)
                )

                if can_compute:
                    predicted_val = sum(
                        coeffs[j] * val_map[missing_idx - j - 1]
                        for j in range(order)
                    )

                    if abs(predicted_val - round(predicted_val)) < 0.001:
                        predicted_val = int(round(predicted_val))

                    coeff_str = ", ".join(f"{c:.2f}" for c in coeffs)
                    predictions[missing_idx] = Prediction(
                        value=predicted_val,
                        confidence=confidence,
                        method=self.name,
                        pattern_name=f"Linear Recurrence (order {order})",
                        explanation=f"Linear recurrence with coefficients [{coeff_str}]"
                    )

            return predictions

        except Exception:
            return {}
