"""Autoencoder-based anomaly detection for trajectory data.

Uses a deep autoencoder to learn a compressed representation of normal
trajectory features. Anomalies are detected as points with high
reconstruction error.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field


class TrajectoryAutoencoder(nn.Module):
    """Deep autoencoder for trajectory feature reconstruction.

    Architecture:
        Encoder: input -> 64 -> 32 -> latent_dim
        Decoder: latent_dim -> 32 -> 64 -> input

    Uses batch normalization and dropout for regularization.
    """

    def __init__(self, input_dim: int, latent_dim: int = 12, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


@dataclass
class AutoencoderDetector:
    """Trains an autoencoder and uses reconstruction error for detection.

    Attributes:
        latent_dim: Size of the bottleneck layer.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        threshold_percentile: Percentile of training errors to use as threshold.
        device: Torch device string.
    """

    latent_dim: int = 12
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    threshold_percentile: float = 95.0
    device: str = "cpu"
    model: TrajectoryAutoencoder | None = field(default=None, init=False, repr=False)
    threshold: float = field(default=0.0, init=False)
    feature_mean: np.ndarray | None = field(default=None, init=False, repr=False)
    feature_std: np.ndarray | None = field(default=None, init=False, repr=False)

    def fit(self, X_train: np.ndarray) -> AutoencoderDetector:
        """Train the autoencoder on normal data.

        Args:
            X_train: Feature matrix of normal samples, shape (n_samples, n_features).

        Returns:
            self for method chaining.
        """
        # Normalize
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X_norm = (X_train - self.feature_mean) / self.feature_std

        tensor = torch.tensor(X_norm, dtype=torch.float32)
        dataset = TensorDataset(tensor, tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_train.shape[1]
        self.model = TrajectoryAutoencoder(input_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

        # Set threshold from training reconstruction errors
        train_errors = self._compute_errors(X_norm)
        self.threshold = float(np.percentile(train_errors, self.threshold_percentile))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1=anomaly, 0=normal).

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Binary predictions array.
        """
        scores = self.score_samples(X)
        return (scores > self.threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (reconstruction error).

        Higher values indicate more anomalous samples.
        """
        X_norm = (X - self.feature_mean) / self.feature_std
        return self._compute_errors(X_norm)

    def _compute_errors(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute per-sample mean squared reconstruction error."""
        self.model.eval()
        tensor = torch.tensor(X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(tensor).cpu().numpy()
        errors = np.mean((X_norm - recon) ** 2, axis=1)
        return errors
