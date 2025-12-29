"""OEIS (Online Encyclopedia of Integer Sequences) integration."""

import asyncio
import re
from typing import Optional
import httpx

from .base import Detector, DetectionResult, Prediction, Sequence


class OEISDetector(Detector):
    """
    Detector that queries the OEIS database for matching sequences.

    The OEIS contains over 350,000 integer sequences and is the definitive
    reference for number sequences.
    """

    name = "oeis"
    BASE_URL = "https://oeis.org/search"

    def __init__(self, timeout: float = 10.0, max_results: int = 5):
        """
        Initialize the OEIS detector.

        Args:
            timeout: HTTP request timeout in seconds
            max_results: Maximum number of OEIS results to consider
        """
        self.timeout = timeout
        self.max_results = max_results

    async def detect(self, sequence: Sequence) -> DetectionResult:
        """Query OEIS and predict missing values."""
        result = DetectionResult()

        # Get known values for query
        known_values = sequence.known_values
        if len(known_values) < 3:
            return result  # Not enough values for reliable OEIS lookup

        # Build search query
        query = ",".join(str(int(v)) for v in known_values if v == int(v))

        try:
            matches = await self._search_oeis(query)

            for match in matches[:self.max_results]:
                predictions = self._extract_predictions(sequence, match)
                for idx, pred in predictions.items():
                    result.add_prediction(idx, pred)

        except Exception as e:
            # OEIS unavailable, return empty result
            pass

        return result

    async def _search_oeis(self, query: str) -> list[dict]:
        """
        Search OEIS for sequences matching the query.

        Args:
            query: Comma-separated sequence values

        Returns:
            List of matching sequence info dicts
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # OEIS search endpoint
            params = {
                "q": query,
                "fmt": "text",
            }

            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()

            return self._parse_oeis_response(response.text)

    def _parse_oeis_response(self, text: str) -> list[dict]:
        """
        Parse OEIS text response into structured data.

        Args:
            text: Raw OEIS response text

        Returns:
            List of parsed sequence information
        """
        results = []
        current_seq = {}

        for line in text.split("\n"):
            line = line.strip()

            # Sequence ID line: %I A000045
            if line.startswith("%I "):
                if current_seq:
                    results.append(current_seq)
                seq_id = line.split()[1] if len(line.split()) > 1 else None
                current_seq = {"id": seq_id, "values": [], "name": ""}

            # Name line: %N A000045 Fibonacci numbers
            elif line.startswith("%N ") and current_seq:
                name_match = re.search(r"%N \w+ (.+)", line)
                if name_match:
                    current_seq["name"] = name_match.group(1)

            # Sequence values: %S A000045 0,1,1,2,3,5,8,13,21,34,55,89,144
            elif line.startswith("%S ") and current_seq:
                values_match = re.search(r"%S \w+ ([\d,\-]+)", line)
                if values_match:
                    values_str = values_match.group(1)
                    try:
                        values = [int(v) for v in values_str.split(",")]
                        current_seq["values"] = values
                    except ValueError:
                        pass

            # Additional values: %T, %U lines
            elif (line.startswith("%T ") or line.startswith("%U ")) and current_seq:
                values_match = re.search(r"%[TU] \w+ ([\d,\-]+)", line)
                if values_match:
                    values_str = values_match.group(1)
                    try:
                        values = [int(v) for v in values_str.split(",")]
                        current_seq["values"].extend(values)
                    except ValueError:
                        pass

        if current_seq:
            results.append(current_seq)

        return results

    def _extract_predictions(
        self, sequence: Sequence, oeis_match: dict
    ) -> dict[int, Prediction]:
        """
        Extract predictions for missing values from an OEIS match.

        Args:
            sequence: The input sequence with missing values
            oeis_match: Matched OEIS sequence info

        Returns:
            Dictionary mapping missing indices to predictions
        """
        predictions = {}
        oeis_values = oeis_match.get("values", [])
        oeis_name = oeis_match.get("name", "Unknown sequence")
        oeis_id = oeis_match.get("id", "")

        if not oeis_values:
            return predictions

        # Find the alignment between input sequence and OEIS sequence
        alignment = self._find_alignment(sequence, oeis_values)

        if alignment is None:
            return predictions

        offset, confidence = alignment

        # Predict missing values
        for missing_idx in sequence.missing_indices:
            oeis_idx = missing_idx + offset

            if 0 <= oeis_idx < len(oeis_values):
                predictions[missing_idx] = Prediction(
                    value=oeis_values[oeis_idx],
                    confidence=confidence,
                    method=self.name,
                    pattern_name=oeis_id,
                    explanation=f"OEIS {oeis_id}: {oeis_name}"
                )

        return predictions

    def _find_alignment(
        self, sequence: Sequence, oeis_values: list[int]
    ) -> Optional[tuple[int, float]]:
        """
        Find the best alignment between input sequence and OEIS values.

        Args:
            sequence: Input sequence
            oeis_values: OEIS sequence values

        Returns:
            Tuple of (offset, confidence) or None if no good alignment
        """
        known_pairs = sequence.known_pairs
        if not known_pairs:
            return None

        best_offset = None
        best_matches = 0
        total_known = len(known_pairs)

        # Try different offsets
        for offset in range(-len(oeis_values), len(sequence.values) + len(oeis_values)):
            matches = 0

            for seq_idx, seq_val in known_pairs:
                oeis_idx = seq_idx + offset

                if 0 <= oeis_idx < len(oeis_values):
                    if abs(oeis_values[oeis_idx] - seq_val) < 0.001:
                        matches += 1

            if matches > best_matches:
                best_matches = matches
                best_offset = offset

        if best_offset is None or best_matches < 2:
            return None

        confidence = best_matches / total_known
        return (best_offset, confidence)


class CachedOEISDetector(OEISDetector):
    """
    OEIS detector with local caching for frequently accessed sequences.

    Caches results to reduce API calls and improve offline capability.
    """

    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self._cache: dict[str, list[dict]] = {}

    async def _search_oeis(self, query: str) -> list[dict]:
        """Search with caching."""
        if query in self._cache:
            return self._cache[query]

        results = await super()._search_oeis(query)
        self._cache[query] = results
        return results
