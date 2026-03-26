"""LSTM-based trajectory prediction model for anomaly detection.

Uses a sequence-to-one LSTM to predict the next position/state given
a window of past observations. High prediction error signals anomalous
trajectory segments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field


class TrajectoryLSTM(nn.Module):
    """LSTM network for trajectory sequence classification.

    Architecture:
        LSTM (bidirectional) -> attention pooling -> FC -> sigmoid

    Learns to distinguish normal from anomalous trajectory windows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim).

        Returns:
            Anomaly logits of shape (batch, 1).
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        return self.classifier(context)


@dataclass
class LSTMDetector:
    """Trains an LSTM classifier for trajectory anomaly detection.

    Attributes:
        hidden_dim: LSTM hidden state dimension.
        n_layers: Number of LSTM layers.
        epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        device: Torch device string.
    """

    hidden_dim: int = 64
    n_layers: int = 2
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    device: str = "cpu"
    model: TrajectoryLSTM | None = field(default=None, init=False, repr=False)
    _seq_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _seq_std: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(
        self, X_sequences: np.ndarray, y_labels: np.ndarray
    ) -> LSTMDetector:
        """Train the LSTM on labeled sequence data.

        Args:
            X_sequences: Shape (n_samples, seq_len, n_features).
            y_labels: Binary labels, shape (n_samples,).

        Returns:
            self for method chaining.
        """
        # Normalize across all timesteps
        flat = X_sequences.reshape(-1, X_sequences.shape[-1])
        self._seq_mean = flat.mean(axis=0)
        self._seq_std = flat.std(axis=0)
        self._seq_std[self._seq_std < 1e-8] = 1.0

        X_norm = (X_sequences - self._seq_mean) / self._seq_std

        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        y_tensor = torch.tensor(y_labels, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_sequences.shape[-1]
        self.model = TrajectoryLSTM(
            input_dim, self.hidden_dim, self.n_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Handle class imbalance with weighted BCE
        n_pos = y_labels.sum()
        n_neg = len(y_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

        return self

    def predict(self, X_sequences: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict anomaly labels for sequences.

        Args:
            X_sequences: Shape (n_samples, seq_len, n_features).
            threshold: Classification threshold on sigmoid output.

        Returns:
            Binary predictions array.
        """
        scores = self.score_samples(X_sequences)
        return (scores > threshold).astype(int)

    def score_samples(self, X_sequences: np.ndarray) -> np.ndarray:
        """Compute anomaly probability scores.

        Returns sigmoid probabilities; higher = more anomalous.
        """
        X_norm = (X_sequences - self._seq_mean) / self._seq_std
        tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
