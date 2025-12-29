"""Transformer model for sequence prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class NumberEmbedding(nn.Module):
    """
    Embedding layer for numbers.

    Since numbers are continuous and unbounded, we use a learned projection
    combined with magnitude encoding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Project scalar to embedding dimension
        self.linear = nn.Linear(1, d_model // 2)

        # Magnitude encoding (log scale)
        self.magnitude_proj = nn.Linear(1, d_model // 2)

        # Combine projections
        self.combine = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len] containing numbers
            mask: Tensor of shape [batch_size, seq_len] where True = missing

        Returns:
            Tensor of shape [batch_size, seq_len, d_model]
        """
        # Expand for linear layer
        x_expanded = x.unsqueeze(-1)  # [batch, seq, 1]

        # Linear projection
        linear_emb = self.linear(x_expanded)  # [batch, seq, d_model//2]

        # Magnitude encoding (log of absolute value + 1)
        magnitude = torch.log1p(torch.abs(x_expanded))
        magnitude_emb = self.magnitude_proj(magnitude)  # [batch, seq, d_model//2]

        # Combine
        combined = torch.cat([linear_emb, magnitude_emb], dim=-1)  # [batch, seq, d_model]
        output = self.combine(combined)

        # Zero out masked positions (missing values)
        output = output * (~mask).unsqueeze(-1).float()

        return output


class SequenceTransformer(nn.Module):
    """
    Transformer encoder for sequence prediction.

    Uses masked language modeling approach - predict the value at masked positions
    given the surrounding context.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_value: int = 10000,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_value = max_value

        # Number embedding
        self.number_embedding = NumberEmbedding(d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection - predict the number value
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_confidence: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, seq_len] with number values
            mask: Boolean tensor [batch_size, seq_len] where True = masked/missing
            return_confidence: Whether to return confidence scores

        Returns:
            predictions: Predicted values [batch_size, seq_len]
            confidence: Confidence scores [batch_size, seq_len] (if requested)
        """
        # Embed numbers
        embedded = self.number_embedding(x, mask)  # [batch, seq, d_model]

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Create attention mask (masked positions can attend to unmasked)
        # We use the standard transformer without attention masking here
        # since we want bidirectional context

        # Transformer encoding
        encoded = self.transformer_encoder(embedded)  # [batch, seq, d_model]

        # Predict values at all positions
        predictions = self.output_proj(encoded).squeeze(-1)  # [batch, seq]

        if return_confidence:
            confidence = self.confidence_head(encoded).squeeze(-1)  # [batch, seq]
            return predictions, confidence
        else:
            return predictions, None

    def predict_masked(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict values only at masked positions.

        Args:
            x: Input tensor with placeholder values at masked positions
            mask: Boolean tensor where True = position to predict

        Returns:
            predictions: Values predicted for masked positions
            confidences: Confidence scores for predictions
        """
        self.eval()
        with torch.no_grad():
            all_preds, all_conf = self.forward(x, mask, return_confidence=True)

            # Extract only masked positions
            masked_preds = all_preds[mask]
            masked_conf = all_conf[mask] if all_conf is not None else torch.ones_like(masked_preds)

            return masked_preds, masked_conf


def create_transformer_model(
    model_size: str = "small",
) -> SequenceTransformer:
    """
    Create a transformer model with preset configurations.

    Args:
        model_size: One of "tiny", "small", "medium", "large"

    Returns:
        Configured SequenceTransformer model
    """
    configs = {
        "tiny": {
            "d_model": 64,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 128,
        },
        "small": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 512,
        },
        "medium": {
            "d_model": 256,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 1024,
        },
        "large": {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 8,
            "dim_feedforward": 2048,
        },
    }

    config = configs.get(model_size, configs["small"])
    return SequenceTransformer(**config)
