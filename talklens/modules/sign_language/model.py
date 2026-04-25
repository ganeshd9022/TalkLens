"""
Sign Language Recognition – LSTM Classifier
─────────────────────────────────────────────
Architecture: Sequence of 63-dimensional landmark vectors → BiLSTM → FC → Softmax
Input:  (batch, seq_len=30, 63)   21 hand points × (x,y,z)
Output: (batch, num_classes)      26 ASL letters + extra gesture classes
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkLSTM(nn.Module):
    """
    Bidirectional LSTM that classifies ASL hand gestures from landmark sequences.
    Designed for real-time CPU inference (< 5 ms per forward pass on modern hardware).
    """

    def __init__(
        self,
        input_size: int   = 63,
        hidden_size: int  = 128,
        num_layers: int   = 2,
        num_classes: int  = 26,
        dropout: float    = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: (batch, seq_len, input_size)"""
        out, _ = self.lstm(x)            # → (batch, seq_len, hidden*2)
        out    = self.layer_norm(out)
        pooled = out.mean(dim=1)         # global average pooling over time
        pooled = self.dropout(pooled)
        return self.classifier(pooled)   # → (batch, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)
