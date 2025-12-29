"""Deep learning-based sequence detector using Transformer and LSTM models."""

import os
import warnings
from pathlib import Path
from typing import Optional
import torch

# Suppress std() warnings from untrained models
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

from .base import Detector, DetectionResult, Prediction, Sequence
from ..models.transformer import SequenceTransformer, create_transformer_model
from ..models.lstm import SequenceLSTM, create_lstm_model


class MLDetector(Detector):
    """
    Machine learning detector that uses neural networks for sequence prediction.

    Combines predictions from both Transformer and LSTM models for better
    coverage of different pattern types.
    """

    name = "ml"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_transformer: bool = True,
        use_lstm: bool = True,
    ):
        """
        Initialize the ML detector.

        Args:
            model_dir: Directory containing pre-trained models
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            use_transformer: Whether to use the Transformer model
            use_lstm: Whether to use the LSTM model
        """
        self.model_dir = Path(model_dir) if model_dir else self._default_model_dir()
        self.device = device or self._get_device()
        self.use_transformer = use_transformer
        self.use_lstm = use_lstm

        self.transformer: Optional[SequenceTransformer] = None
        self.lstm: Optional[SequenceLSTM] = None

        self._load_models()

    def _default_model_dir(self) -> Path:
        """Get default model directory."""
        return Path.home() / ".seq-analyzer" / "models"

    def _get_device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_models(self) -> None:
        """Load pre-trained models if available, otherwise create new ones."""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        if self.use_transformer:
            transformer_path = self.model_dir / "transformer.pt"
            if transformer_path.exists():
                self.transformer = create_transformer_model("small")
                self.transformer.load_state_dict(torch.load(transformer_path, map_location=self.device))
            else:
                # Create untrained model (will need training)
                self.transformer = create_transformer_model("small")

            self.transformer = self.transformer.to(self.device)
            self.transformer.eval()

        if self.use_lstm:
            lstm_path = self.model_dir / "lstm.pt"
            if lstm_path.exists():
                self.lstm = create_lstm_model("small")
                self.lstm.load_state_dict(torch.load(lstm_path, map_location=self.device))
            else:
                # Create untrained model
                self.lstm = create_lstm_model("small")

            self.lstm = self.lstm.to(self.device)
            self.lstm.eval()

    async def detect(self, sequence: Sequence) -> DetectionResult:
        """Run ML models to predict missing values."""
        result = DetectionResult()

        # Prepare input tensors
        values, mask = self._prepare_input(sequence)

        # Get predictions from each model
        if self.transformer is not None:
            transformer_preds = self._predict_with_model(
                self.transformer, values, mask, "transformer"
            )
            for idx, pred in transformer_preds.items():
                result.add_prediction(idx, pred)

        if self.lstm is not None:
            lstm_preds = self._predict_with_model(
                self.lstm, values, mask, "lstm"
            )
            for idx, pred in lstm_preds.items():
                result.add_prediction(idx, pred)

        return result

    def _prepare_input(self, sequence: Sequence) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare sequence for model input.

        Args:
            sequence: Input sequence

        Returns:
            Tuple of (values tensor, mask tensor)
        """
        # Convert to tensor, using 0 as placeholder for missing values
        values = []
        mask = []

        for v in sequence.values:
            if v is None:
                values.append(0.0)
                mask.append(True)
            else:
                values.append(float(v))
                mask.append(False)

        values_tensor = torch.tensor([values], dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.bool, device=self.device)

        return values_tensor, mask_tensor

    def _predict_with_model(
        self,
        model: torch.nn.Module,
        values: torch.Tensor,
        mask: torch.Tensor,
        model_name: str,
    ) -> dict[int, Prediction]:
        """
        Get predictions from a single model.

        Args:
            model: The neural network model
            values: Input values tensor
            mask: Mask tensor
            model_name: Name for logging

        Returns:
            Dictionary mapping indices to predictions
        """
        predictions = {}

        try:
            with torch.no_grad():
                all_preds, all_conf = model(values, mask, return_confidence=True)

                # Extract predictions for masked positions
                all_preds = all_preds.squeeze(0)  # Remove batch dim
                all_conf = all_conf.squeeze(0) if all_conf is not None else torch.ones_like(all_preds)

                mask_1d = mask.squeeze(0)

                for idx in range(len(mask_1d)):
                    if mask_1d[idx]:
                        pred_value = all_preds[idx].item()
                        confidence = all_conf[idx].item()

                        # Round if close to integer
                        if abs(pred_value - round(pred_value)) < 0.1:
                            pred_value = int(round(pred_value))

                        # Reduce confidence for untrained models
                        if not self._is_trained(model):
                            confidence *= 0.3  # Untrained model penalty

                        predictions[idx] = Prediction(
                            value=pred_value,
                            confidence=confidence,
                            method=f"{self.name}/{model_name}",
                            pattern_name=f"Neural ({model_name})",
                            explanation=f"Predicted by {model_name} model"
                        )

        except Exception as e:
            # Model inference failed
            pass

        return predictions

    def _is_trained(self, model: torch.nn.Module) -> bool:
        """Check if a model has been trained (heuristic)."""
        # Simple heuristic: check if weights are non-random
        # This is approximate - a proper solution would track training state
        for param in model.parameters():
            if param.requires_grad:
                std = param.std().item()
                # Random init typically has std around 0.5-1.0
                # Trained models often have different distributions
                if std < 0.01 or std > 2.0:
                    return True
        return False

    def save_models(self) -> None:
        """Save models to disk."""
        if self.transformer is not None:
            torch.save(
                self.transformer.state_dict(),
                self.model_dir / "transformer.pt"
            )

        if self.lstm is not None:
            torch.save(
                self.lstm.state_dict(),
                self.model_dir / "lstm.pt"
            )


class TrainableMLDetector(MLDetector):
    """
    ML Detector with training capabilities.

    Can be trained on OEIS data or custom sequence datasets.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimizer_transformer = None
        self.optimizer_lstm = None

    def setup_training(self, lr: float = 1e-4):
        """Set up optimizers for training."""
        if self.transformer is not None:
            self.transformer.train()
            self.optimizer_transformer = torch.optim.AdamW(
                self.transformer.parameters(), lr=lr
            )

        if self.lstm is not None:
            self.lstm.train()
            self.optimizer_lstm = torch.optim.AdamW(
                self.lstm.parameters(), lr=lr
            )

    def train_step(
        self,
        sequences: list[list[float]],
        mask_ratio: float = 0.15,
    ) -> dict[str, float]:
        """
        Perform one training step on a batch of sequences.

        Args:
            sequences: List of complete sequences (no missing values)
            mask_ratio: Fraction of positions to mask

        Returns:
            Dictionary of loss values
        """
        losses = {}

        # Prepare batch
        max_len = max(len(s) for s in sequences)
        batch_values = torch.zeros(len(sequences), max_len, device=self.device)
        batch_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool, device=self.device)
        batch_targets = torch.zeros(len(sequences), max_len, device=self.device)

        for i, seq in enumerate(sequences):
            for j, v in enumerate(seq):
                batch_values[i, j] = v
                batch_targets[i, j] = v

            # Random masking
            n_mask = max(1, int(len(seq) * mask_ratio))
            mask_indices = torch.randperm(len(seq))[:n_mask]
            batch_mask[i, mask_indices] = True
            batch_values[i, mask_indices] = 0  # Replace masked with 0

        # Train Transformer
        if self.transformer is not None and self.optimizer_transformer is not None:
            self.optimizer_transformer.zero_grad()
            preds, _ = self.transformer(batch_values, batch_mask, return_confidence=False)
            loss = torch.nn.functional.mse_loss(preds[batch_mask], batch_targets[batch_mask])
            loss.backward()
            self.optimizer_transformer.step()
            losses["transformer"] = loss.item()

        # Train LSTM
        if self.lstm is not None and self.optimizer_lstm is not None:
            self.optimizer_lstm.zero_grad()
            preds, _ = self.lstm(batch_values, batch_mask, return_confidence=False)
            loss = torch.nn.functional.mse_loss(preds[batch_mask], batch_targets[batch_mask])
            loss.backward()
            self.optimizer_lstm.step()
            losses["lstm"] = loss.item()

        return losses
