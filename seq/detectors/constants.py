"""Detector for sequences based on famous mathematical constants."""

from typing import Optional
from .base import Detector, DetectionResult, Prediction, Sequence


# Digits of famous mathematical constants (first 100+ digits each)
MATHEMATICAL_CONSTANTS = {
    "e (Euler's number)": (
        "2.71828182845904523536028747135266249775724709369995"
        "95749669676277240766303535475945713821785251664274"
    ),
    "pi": (
        "3.14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
    ),
    "phi (Golden Ratio)": (
        "1.61803398874989484820458683436563811772030917980576"
        "28621354486227052604628189024497072072041893911374"
    ),
    "sqrt(2)": (
        "1.41421356237309504880168872420969807856967187537694"
        "80731766797379907324784621070388503875343276415727"
    ),
    "sqrt(3)": (
        "1.73205080756887729352744634150587236694280525381038"
        "06280558069794519330169088000370811461867572485756"
    ),
    "sqrt(5)": (
        "2.23606797749978969640917366873127623544061835961152"
        "57242708972454105209256378048994144144083787822749"
    ),
    "ln(2)": (
        "0.69314718055994530941723212145817656807550013436025"
        "52541206800094933936219696947156058633269964186875"
    ),
    "Euler-Mascheroni gamma": (
        "0.57721566490153286060651209008240243104215933593992"
        "35988057672348848677267776646709369470632917467495"
    ),
    "Catalan's constant": (
        "0.91596559417721901505460351493238411077414937428167"
        "21342664981196217630197762547694793565129261151062"
    ),
    "Apery's constant (zeta(3))": (
        "1.20205690315959428539973816151144999076498629234049"
        "88817922715553418382057863130901864558736093352581"
    ),
}


def _extract_digits(constant_str: str) -> list[int]:
    """Extract digits from a constant string (ignoring decimal point)."""
    return [int(c) for c in constant_str if c.isdigit()]


def _find_subsequence_match(
    known_values: list[int],
    known_indices: list[int],
    constant_digits: list[int],
    max_offset: int = 50,
) -> Optional[tuple[int, float]]:
    """
    Find if known values match a subsequence of constant digits.

    Args:
        known_values: The known integer values from the sequence
        known_indices: The positions of known values in the original sequence
        constant_digits: Digits of the mathematical constant
        max_offset: Maximum offset to search for alignment

    Returns:
        Tuple of (offset, confidence) if match found, None otherwise
    """
    if not known_values or not constant_digits:
        return None

    # The first known value's position in the original sequence
    first_known_idx = known_indices[0]

    # Try different offsets (where in the constant does our sequence start?)
    for offset in range(max_offset):
        matches = 0
        total = 0

        for seq_idx, value in zip(known_indices, known_values):
            # Position in the constant digits
            const_pos = offset + (seq_idx - first_known_idx)

            if const_pos < 0 or const_pos >= len(constant_digits):
                # Out of bounds - this offset doesn't work
                break

            total += 1
            if constant_digits[const_pos] == value:
                matches += 1

        if total > 0 and matches == total:
            # Perfect match at this offset
            confidence = min(0.95, 0.7 + 0.05 * matches)  # Higher with more matches
            return (offset, confidence)

    return None


class ConstantsDetector(Detector):
    """
    Detector that identifies sequences as digits of famous mathematical constants.

    Supports:
    - e (Euler's number): 2.71828...
    - pi: 3.14159...
    - phi (Golden Ratio): 1.61803...
    - sqrt(2), sqrt(3), sqrt(5)
    - ln(2)
    - Euler-Mascheroni constant
    - Catalan's constant
    - Apery's constant (zeta(3))
    """

    name = "constants"

    def __init__(self):
        """Initialize with precomputed digit lists."""
        self.constants = {
            name: _extract_digits(value)
            for name, value in MATHEMATICAL_CONSTANTS.items()
        }

    async def detect(self, sequence: Sequence) -> DetectionResult:
        """Detect if sequence matches digits of a mathematical constant."""
        result = DetectionResult()

        # Get known values - this detector only works with single-digit integers
        known_pairs = sequence.known_pairs
        if not known_pairs:
            return result

        # Check if all known values are single-digit integers (0-9)
        known_values = []
        known_indices = []
        for idx, val in known_pairs:
            if not isinstance(val, (int, float)):
                return result
            if isinstance(val, float) and val != int(val):
                return result  # Not an integer
            int_val = int(val)
            if int_val < 0 or int_val > 9:
                return result  # Not a single digit
            known_values.append(int_val)
            known_indices.append(idx)

        if len(known_values) < 3:
            return result  # Need at least 3 digits for meaningful match

        # Try to match against each constant
        for const_name, const_digits in self.constants.items():
            match_result = _find_subsequence_match(
                known_values, known_indices, const_digits
            )

            if match_result is not None:
                offset, confidence = match_result
                first_known_idx = known_indices[0]

                # Generate predictions for missing values
                for missing_idx in sequence.missing_indices:
                    const_pos = offset + (missing_idx - first_known_idx)

                    if 0 <= const_pos < len(const_digits):
                        predicted_value = const_digits[const_pos]

                        result.add_prediction(
                            missing_idx,
                            Prediction(
                                value=predicted_value,
                                confidence=confidence,
                                method=self.name,
                                pattern_name=f"Digits of {const_name}",
                                explanation=f"Digits of {const_name} starting at position {offset + 1}"
                            )
                        )

        return result
