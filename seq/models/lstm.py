"""Bidirectional LSTM model for sequence prediction."""

import torch
import torch.nn as nn
from typing import Optional


class NumberEncoder(nn.Module):
    """Encode numbers into a dense representation."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Multi-scale encoding for numbers
        self.linear = nn.Linear(1, hidden_size // 2)
        self.log_linear = nn.Linear(1, hidden_size // 4)
        self.sign_linear = nn.Linear(1, hidden_size // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [batch, seq] of number values

        Returns:
            Encoded tensor [batch, seq, hidden_size]
        """
        x = x.unsqueeze(-1)  # [batch, seq, 1]

        # Linear encoding
        lin = self.linear(x)  # [batch, seq, hidden//2]

        # Log magnitude encoding
        log_mag = torch.log1p(torch.abs(x))
        log_enc = self.log_linear(log_mag)  # [batch, seq, hidden//4]

        # Sign encoding
        sign = torch.sign(x)
        sign_enc = self.sign_linear(sign)  # [batch, seq, hidden//4]

        # Concatenate
        return torch.cat([lin, log_enc, sign_enc], dim=-1)


class SequenceLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence prediction.

    Uses bidirectional processing to capture context from both directions,
    which is important for predicting missing values in the middle of sequences.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Number encoder
        self.encoder = NumberEncoder(hidden_size)

        # Mask embedding (learnable representation for missing values)
        self.mask_embedding = nn.Parameter(torch.randn(hidden_size))

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
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
        batch_size, seq_len = x.shape

        # Encode numbers
        encoded = self.encoder(x)  # [batch, seq, hidden]

        # Replace masked positions with learnable mask embedding
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        mask_emb = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        encoded = torch.where(mask_expanded, mask_emb, encoded)

        # LSTM processing
        lstm_out, _ = self.lstm(encoded)  # [batch, seq, hidden*2]

        # Predict values
        predictions = self.output_proj(lstm_out).squeeze(-1)  # [batch, seq]

        if return_confidence:
            confidence = self.confidence_head(lstm_out).squeeze(-1)
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


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for better context aggregation.

    Adds an attention layer on top of the LSTM to focus on relevant
    positions when making predictions.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()

        self.base_lstm = SequenceLSTM(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Override output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_confidence: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward with attention."""
        batch_size, seq_len = x.shape

        # Get LSTM output (before output projection)
        encoded = self.base_lstm.encoder(x)
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        mask_emb = self.base_lstm.mask_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        encoded = torch.where(mask_expanded, mask_emb, encoded)
        lstm_out, _ = self.base_lstm.lstm(encoded)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual + layer norm
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Output
        predictions = self.output_proj(attn_out).squeeze(-1)

        if return_confidence:
            confidence = self.confidence_head(attn_out).squeeze(-1)
            return predictions, confidence
        else:
            return predictions, None


def create_lstm_model(
    model_size: str = "small",
    use_attention: bool = False,
) -> nn.Module:
    """
    Create an LSTM model with preset configurations.

    Args:
        model_size: One of "tiny", "small", "medium", "large"
        use_attention: Whether to use attention-enhanced LSTM

    Returns:
        Configured LSTM model
    """
    configs = {
        "tiny": {"hidden_size": 64, "num_layers": 1},
        "small": {"hidden_size": 128, "num_layers": 2},
        "medium": {"hidden_size": 256, "num_layers": 3},
        "large": {"hidden_size": 512, "num_layers": 4},
    }

    config = configs.get(model_size, configs["small"])

    if use_attention:
        return AttentionLSTM(**config)
    else:
        return SequenceLSTM(**config)
