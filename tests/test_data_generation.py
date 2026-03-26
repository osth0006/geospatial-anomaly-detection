"""Tests for the trajectory data generation module."""

import numpy as np
import pandas as pd
import pytest

from src.data_generation.generator import TrajectoryGenerator, RouteConfig
from src.data_generation.anomalies import AnomalyInjector


class TestTrajectoryGenerator:
    def setup_method(self):
        self.gen = TrajectoryGenerator(seed=42)

    def test_generates_correct_number_of_tracks(self):
        df = self.gen.generate_dataset(n_tracks=20)
        assert df["track_id"].nunique() == 20

    def test_output_columns(self):
        df = self.gen.generate_dataset(n_tracks=5)
        expected = {
            "track_id", "timestamp", "latitude", "longitude",
            "speed_knots", "heading_deg", "entity_type", "route_name",
        }
        assert set(df.columns) == expected

    def test_coordinates_in_valid_range(self):
        df = self.gen.generate_dataset(n_tracks=30)
        assert df["latitude"].between(-90, 90).all()
        assert df["longitude"].between(-180, 180).all()

    def test_speed_is_positive(self):
        df = self.gen.generate_dataset(n_tracks=10)
        assert (df["speed_knots"] > 0).all()

    def test_headings_in_range(self):
        df = self.gen.generate_dataset(n_tracks=10)
        assert df["heading_deg"].between(0, 360).all()

    def test_timestamps_are_monotonic_per_track(self):
        df = self.gen.generate_dataset(n_tracks=10)
        for _, group in df.groupby("track_id"):
            assert group["timestamp"].is_monotonic_increasing

    def test_reproducibility(self):
        gen1 = TrajectoryGenerator(seed=123)
        gen2 = TrajectoryGenerator(seed=123)
        df1 = gen1.generate_dataset(n_tracks=5)
        df2 = gen2.generate_dataset(n_tracks=5)
        pd.testing.assert_frame_equal(df1, df2)

    def test_haversine_known_distance(self):
        # NYC to London is roughly 5570 km
        nyc = np.array([40.7128, -74.0060])
        london = np.array([51.5074, -0.1278])
        dist = TrajectoryGenerator._haversine_km(nyc, london)
        assert 5500 < dist < 5700


class TestAnomalyInjector:
    def setup_method(self):
        gen = TrajectoryGenerator(seed=42)
        self.df = gen.generate_dataset(n_tracks=50)
        self.injector = AnomalyInjector(anomaly_fraction=0.2, seed=42)

    def test_adds_anomaly_columns(self):
        result = self.injector.inject(self.df)
        assert "is_anomaly" in result.columns
        assert "anomaly_type" in result.columns

    def test_anomaly_fraction(self):
        result = self.injector.inject(self.df)
        anomalous_tracks = result[result["is_anomaly"]]["track_id"].nunique()
        expected = int(50 * 0.2)
        assert anomalous_tracks == expected

    def test_normal_points_labeled(self):
        result = self.injector.inject(self.df)
        normal = result[~result["is_anomaly"]]
        assert (normal["anomaly_type"] == "normal").all()

    def test_anomaly_types_valid(self):
        result = self.injector.inject(self.df)
        valid_types = {"normal", "route_deviation", "speed_anomaly", "loitering", "dark_period"}
        assert set(result["anomaly_type"].unique()).issubset(valid_types)

    def test_preserves_data_shape(self):
        result = self.injector.inject(self.df)
        assert len(result) == len(self.df)
        assert result["track_id"].nunique() == self.df["track_id"].nunique()
