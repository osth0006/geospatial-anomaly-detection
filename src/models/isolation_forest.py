"""Isolation Forest baseline for trajectory anomaly detection.

Provides a scikit-learn-based baseline detector using Isolation Forest,
which isolates anomalies by random recursive partitioning.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field


@dataclass
class IsolationForestDetector:
    """Isolation Forest anomaly detector wrapper.

    Wraps scikit-learn's IsolationForest with standardization
    and a consistent interface matching the other detectors.

    Attributes:
        contamination: Expected proportion of anomalies.
        n_estimators: Number of trees in the forest.
        max_samples: Samples drawn per tree.
        seed: Random state for reproducibility.
    """

    contamination: float = 0.1
    n_estimators: int = 200
    max_samples: int | str = "auto"
    seed: int = 42
    _model: IsolationForest | None = field(default=None, init=False, repr=False)
    _scaler: StandardScaler = field(default_factory=StandardScaler, init=False, repr=False)

    def fit(self, X_train: np.ndarray) -> IsolationForestDetector:
        """Fit the Isolation Forest on training data.

        Args:
            X_train: Feature matrix, shape (n_samples, n_features).

        Returns:
            self for method chaining.
        """
        X_scaled = self._scaler.fit_transform(X_train)
        self._model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.seed,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1=anomaly, 0=normal).

        Converts sklearn's convention (-1=anomaly, 1=normal) to (1, 0).
        """
        X_scaled = self._scaler.transform(X)
        preds = self._model.predict(X_scaled)
        return (preds == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores.

        Returns negated decision function so higher = more anomalous,
        consistent with the autoencoder detector.
        """
        X_scaled = self._scaler.transform(X)
        return -self._model.decision_function(X_scaled)
