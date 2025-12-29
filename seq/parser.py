"""Parse sequence strings into Sequence objects."""

import re
from typing import Optional

from .detectors.base import Sequence


def parse_sequence(input_str: str) -> Sequence:
    """
    Parse a string representation of a sequence into a Sequence object.

    Supports various formats:
    - Comma-separated: "1, 2, ?, 4, 5"
    - Space-separated: "1 2 ? 4 5"
    - With brackets: "[1, 2, ?, 4, 5]"
    - Mixed: "1,2,?,4,5"

    Missing values can be represented as:
    - ? or ??
    - _ or __
    - x or X
    - blank between delimiters

    Args:
        input_str: String representation of the sequence

    Returns:
        Sequence object with values and None for missing positions

    Raises:
        ValueError: If the input cannot be parsed
    """
    # Strip brackets if present
    input_str = input_str.strip()
    input_str = re.sub(r'^[\[\(]', '', input_str)
    input_str = re.sub(r'[\]\)]$', '', input_str)

    # Normalize delimiters - replace various separators with comma
    # Handle comma, semicolon, or multiple spaces as delimiters
    normalized = re.sub(r'[,;\s]+', ',', input_str)

    # Split into tokens
    tokens = [t.strip() for t in normalized.split(',') if t.strip() or t == '']

    if not tokens:
        raise ValueError("Empty sequence provided")

    values: list[int | float | None] = []

    for token in tokens:
        parsed = _parse_token(token)
        values.append(parsed)

    if not any(v is not None for v in values):
        raise ValueError("Sequence must have at least one known value")

    return Sequence(values=values)


def _parse_token(token: str) -> Optional[int | float]:
    """
    Parse a single token into a number or None (for missing).

    Args:
        token: String token to parse

    Returns:
        Integer, float, or None for missing values
    """
    token = token.strip()

    # Check for missing value indicators
    missing_patterns = [
        r'^[?]+$',      # ? or ??
        r'^[_]+$',      # _ or __
        r'^[xX]$',      # x or X
        r'^$',          # empty string
        r'^-$',         # dash
        r'^\*+$',       # asterisks
    ]

    for pattern in missing_patterns:
        if re.match(pattern, token):
            return None

    # Try to parse as number
    try:
        # Try integer first
        if '.' not in token and 'e' not in token.lower():
            return int(token)
        # Then try float
        return float(token)
    except ValueError:
        raise ValueError(f"Cannot parse token '{token}' as number or missing value indicator")


def format_sequence(sequence: Sequence, highlight_missing: bool = True) -> str:
    """
    Format a sequence for display.

    Args:
        sequence: The sequence to format
        highlight_missing: Whether to highlight missing values

    Returns:
        Formatted string representation
    """
    parts = []
    for i, v in enumerate(sequence.values):
        if v is None:
            parts.append("[?]" if highlight_missing else "?")
        else:
            # Format integers without decimal, floats with precision
            if isinstance(v, float) and v == int(v):
                parts.append(str(int(v)))
            else:
                parts.append(str(v))

    return ", ".join(parts)
