"""Base detector interface for sequence prediction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prediction:
    """A prediction for a missing value in a sequence."""

    value: int | float
    confidence: float  # 0.0 to 1.0
    method: str  # Name of the detection method
    explanation: Optional[str] = None  # Human-readable explanation
    pattern_name: Optional[str] = None  # e.g., "Fibonacci", "Arithmetic"

    def __repr__(self) -> str:
        return f"Prediction({self.value}, conf={self.confidence:.2f}, method={self.method})"


@dataclass
class Sequence:
    """A sequence with potentially missing values."""

    values: list[int | float | None]  # None represents missing values

    @property
    def known_values(self) -> list[int | float]:
        """Return only the known (non-None) values."""
        return [v for v in self.values if v is not None]

    @property
    def missing_indices(self) -> list[int]:
        """Return indices of missing values."""
        return [i for i, v in enumerate(self.values) if v is None]

    @property
    def known_pairs(self) -> list[tuple[int, int | float]]:
        """Return (index, value) pairs for known values."""
        return [(i, v) for i, v in enumerate(self.values) if v is not None]

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        display = [str(v) if v is not None else "?" for v in self.values]
        return f"Sequence([{', '.join(display)}])"


@dataclass
class DetectionResult:
    """Result from a detector for all missing values."""

    predictions: dict[int, list[Prediction]] = field(default_factory=dict)
    # Maps missing index -> list of predictions (ranked by confidence)

    def add_prediction(self, index: int, prediction: Prediction) -> None:
        """Add a prediction for a missing index."""
        if index not in self.predictions:
            self.predictions[index] = []
        self.predictions[index].append(prediction)

    def get_best(self, index: int) -> Optional[Prediction]:
        """Get the highest confidence prediction for an index."""
        if index not in self.predictions or not self.predictions[index]:
            return None
        return max(self.predictions[index], key=lambda p: p.confidence)


class Detector(ABC):
    """Abstract base class for sequence detectors."""

    name: str = "base"

    @abstractmethod
    async def detect(self, sequence: Sequence) -> DetectionResult:
        """
        Analyze a sequence and predict missing values.

        Args:
            sequence: The sequence with missing values

        Returns:
            DetectionResult with predictions for each missing index
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
