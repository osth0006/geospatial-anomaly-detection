"""Trajectory feature engineering for anomaly detection.

Extracts spatial, temporal, and kinematic features from raw trajectory
data to create feature vectors suitable for ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.data_generation.generator import POINTS_OF_INTEREST


@dataclass
class TrajectoryFeatureExtractor:
    """Extracts engineered features from trajectory DataFrames.

    Computes per-point and per-track features including kinematics,
    proximity to POIs, temporal patterns, and statistical summaries.
    """

    poi_coords: dict[str, tuple[float, float]] | None = None

    def __post_init__(self):
        if self.poi_coords is None:
            self.poi_coords = POINTS_OF_INTEREST

    def extract_point_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-point features for each observation.

        Features added:
        - acceleration: Change in speed between consecutive points
        - heading_change: Angular change in heading
        - hour_of_day: Cyclical time encoding (sin/cos)
        - day_of_week: Cyclical day encoding (sin/cos)
        - dist_to_*: Distance to each point of interest
        - speed_zscore: Z-score of speed within each track
        - heading_change_zscore: Z-score of heading change within track
        """
        df = df.copy()
        df = df.sort_values(["track_id", "timestamp"]).reset_index(drop=True)

        # Kinematics
        df["acceleration"] = df.groupby("track_id")["speed_knots"].diff().fillna(0)
        df["heading_change"] = (
            df.groupby("track_id")["heading_deg"]
            .diff()
            .fillna(0)
            .apply(lambda x: ((x + 180) % 360) - 180)  # Normalize to [-180, 180]
        )
        df["abs_heading_change"] = df["heading_change"].abs()

        # Temporal features (cyclical encoding)
        ts = pd.to_datetime(df["timestamp"])
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        # Proximity to points of interest
        for poi_name, (poi_lat, poi_lon) in self.poi_coords.items():
            df[f"dist_to_{poi_name}"] = self._haversine_vec(
                df["latitude"].values, df["longitude"].values,
                poi_lat, poi_lon,
            )

        # Per-track z-scores
        for col in ["speed_knots", "abs_heading_change"]:
            zscore_col = f"{col}_zscore"
            group_stats = df.groupby("track_id")[col].transform
            mean = group_stats("mean")
            std = group_stats("std").replace(0, 1)
            df[zscore_col] = (df[col] - mean) / std

        return df

    def extract_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-track summary features.

        Returns one row per track with aggregate statistics.
        """
        if "acceleration" not in df.columns:
            df = self.extract_point_features(df)

        agg_dict = {
            "speed_knots": ["mean", "std", "min", "max"],
            "acceleration": ["mean", "std", "max"],
            "abs_heading_change": ["mean", "std", "max"],
            "latitude": ["mean", "std"],
            "longitude": ["mean", "std"],
        }

        # Add POI distance aggregations
        for poi_name in self.poi_coords:
            agg_dict[f"dist_to_{poi_name}"] = ["min", "mean"]

        track_features = df.groupby("track_id").agg(agg_dict)
        track_features.columns = ["_".join(col) for col in track_features.columns]
        track_features = track_features.reset_index()

        # Track-level metadata
        track_meta = df.groupby("track_id").agg(
            n_points=("latitude", "count"),
            duration_hours=("timestamp", lambda x: (x.max() - x.min()).total_seconds() / 3600),
            entity_type=("entity_type", "first"),
            is_anomalous=("is_anomaly", "any"),
        ).reset_index()

        track_features = track_features.merge(track_meta, on="track_id")

        # Total distance traveled
        def total_distance(group):
            lats = group["latitude"].values
            lons = group["longitude"].values
            dists = self._haversine_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
            return dists.sum()

        track_dist = df.groupby("track_id").apply(total_distance, include_groups=False)
        track_dist.name = "total_distance_km"
        track_features = track_features.merge(
            track_dist.reset_index(), on="track_id"
        )

        return track_features

    def get_sequence_data(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        seq_length: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sliding-window sequences for LSTM input.

        Args:
            df: Point-level feature DataFrame.
            feature_cols: Columns to use as features.
            seq_length: Window size for sequences.

        Returns:
            Tuple of (sequences, labels, track_ids) arrays.
        """
        if feature_cols is None:
            feature_cols = [
                "speed_knots", "acceleration", "heading_change",
                "abs_heading_change", "hour_sin", "hour_cos",
                "latitude", "longitude",
            ]

        sequences = []
        labels = []
        tids = []

        for tid, group in df.groupby("track_id"):
            if len(group) < seq_length:
                continue
            vals = group[feature_cols].values
            anomaly = group["is_anomaly"].values

            for i in range(len(vals) - seq_length):
                sequences.append(vals[i : i + seq_length])
                # Label: 1 if any point in the window is anomalous
                labels.append(int(anomaly[i : i + seq_length].any()))
                tids.append(tid)

        return (
            np.array(sequences, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(tids),
        )

    @staticmethod
    def _haversine_vec(
        lat1: np.ndarray, lon1: np.ndarray,
        lat2: float | np.ndarray, lon2: float | np.ndarray,
    ) -> np.ndarray:
        """Vectorized haversine distance in km."""
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2, lon2 = np.radians(lat2), np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
