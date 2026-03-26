"""Tests for the anomaly detection models."""

import numpy as np
import pytest

from src.models.autoencoder import AutoencoderDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.lstm import LSTMDetector


class TestAutoencoderDetector:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X_normal = rng.standard_normal((200, 10)).astype(np.float32)
        X_test = np.vstack([
            rng.standard_normal((50, 10)),
            rng.standard_normal((20, 10)) * 5 + 10,  # Anomalous
        ]).astype(np.float32)

        detector = AutoencoderDetector(
            latent_dim=4, epochs=10, batch_size=64
        )
        detector.fit(X_normal)
        preds = detector.predict(X_test)

        assert preds.shape == (70,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_samples_returns_nonneg(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 8)).astype(np.float32)
        detector = AutoencoderDetector(epochs=5)
        detector.fit(X)
        scores = detector.score_samples(X)
        assert (scores >= 0).all()


class TestIsolationForestDetector:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((200, 8))
        X_test = rng.standard_normal((50, 8))

        detector = IsolationForestDetector(contamination=0.1)
        detector.fit(X_train)
        preds = detector.predict(X_test)

        assert preds.shape == (50,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_samples_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 6))
        detector = IsolationForestDetector()
        detector.fit(X)
        scores = detector.score_samples(X)
        assert scores.shape == (100,)


class TestLSTMDetector:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        n_samples, seq_len, n_features = 100, 20, 6
        X = rng.standard_normal((n_samples, seq_len, n_features)).astype(np.float32)
        y = (rng.random(n_samples) > 0.85).astype(np.int64)

        detector = LSTMDetector(hidden_dim=16, n_layers=1, epochs=5, batch_size=32)
        detector.fit(X, y)
        preds = detector.predict(X)

        assert preds.shape == (n_samples,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_samples_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 15, 4)).astype(np.float32)
        y = np.zeros(50, dtype=np.int64)
        y[:5] = 1

        detector = LSTMDetector(hidden_dim=8, n_layers=1, epochs=3)
        detector.fit(X, y)
        scores = detector.score_samples(X)
        assert (scores >= 0).all() and (scores <= 1).all()
