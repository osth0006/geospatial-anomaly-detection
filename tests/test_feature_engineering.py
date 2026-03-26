"""Tests for the feature engineering module."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation.generator import TrajectoryGenerator
from src.data_generation.anomalies import AnomalyInjector
from src.feature_engineering.features import TrajectoryFeatureExtractor


class TestTrajectoryFeatureExtractor:
    def setup_method(self):
        gen = TrajectoryGenerator(seed=42)
        df = gen.generate_dataset(n_tracks=20)
        injector = AnomalyInjector(anomaly_fraction=0.15, seed=42)
        self.df = injector.inject(df)
        self.extractor = TrajectoryFeatureExtractor()

    def test_point_features_columns(self):
        result = self.extractor.extract_point_features(self.df)
        expected_new = {
            "acceleration", "heading_change", "abs_heading_change",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "speed_knots_zscore", "abs_heading_change_zscore",
        }
        assert expected_new.issubset(set(result.columns))

    def test_point_features_preserves_length(self):
        result = self.extractor.extract_point_features(self.df)
        assert len(result) == len(self.df)

    def test_cyclical_encoding_range(self):
        result = self.extractor.extract_point_features(self.df)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert result[col].between(-1, 1).all()

    def test_track_features_one_row_per_track(self):
        point_df = self.extractor.extract_point_features(self.df)
        track_df = self.extractor.extract_track_features(point_df)
        assert len(track_df) == self.df["track_id"].nunique()

    def test_track_features_has_distance(self):
        point_df = self.extractor.extract_point_features(self.df)
        track_df = self.extractor.extract_track_features(point_df)
        assert "total_distance_km" in track_df.columns
        assert (track_df["total_distance_km"] >= 0).all()

    def test_sequence_data_shape(self):
        point_df = self.extractor.extract_point_features(self.df)
        seq_len = 20
        X, y, tids = self.extractor.get_sequence_data(point_df, seq_length=seq_len)
        assert X.ndim == 3
        assert X.shape[1] == seq_len
        assert len(y) == len(X)
        assert len(tids) == len(X)

    def test_haversine_vec_zero_distance(self):
        dist = TrajectoryFeatureExtractor._haversine_vec(
            np.array([0.0]), np.array([0.0]), 0.0, 0.0,
        )
        assert np.isclose(dist[0], 0.0, atol=1e-6)
