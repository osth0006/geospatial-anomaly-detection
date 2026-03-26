"""Anomaly injection for geospatial trajectory data.

Injects realistic anomalous behaviors into normal trajectory data:
- Route deviations (off-course movements)
- Speed anomalies (sudden acceleration/deceleration)
- Loitering (circling or stopping in unusual locations)
- Dark periods (gaps in position reports, simulating AIS shutoff)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

AnomalyType = Literal[
    "route_deviation", "speed_anomaly", "loitering", "dark_period"
]

ANOMALY_TYPES: list[AnomalyType] = [
    "route_deviation", "speed_anomaly", "loitering", "dark_period"
]


@dataclass
class AnomalyInjector:
    """Injects anomalous behaviors into trajectory data.

    Selects a fraction of tracks and applies one or more anomaly types,
    labeling each point as normal (0) or anomalous (1).

    Attributes:
        anomaly_fraction: Proportion of tracks to make anomalous.
        seed: Random seed for reproducibility.
    """

    anomaly_fraction: float = 0.15
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def inject(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject anomalies into a trajectory dataset.

        Args:
            df: Trajectory DataFrame from TrajectoryGenerator.

        Returns:
            DataFrame with added columns: is_anomaly (bool),
            anomaly_type (str or 'normal').
        """
        df = df.copy()
        df["is_anomaly"] = False
        df["anomaly_type"] = "normal"

        track_ids = df["track_id"].unique()
        n_anomalous = max(1, int(len(track_ids) * self.anomaly_fraction))
        anomalous_ids = self.rng.choice(track_ids, size=n_anomalous, replace=False)

        for tid in anomalous_ids:
            mask = df["track_id"] == tid
            anomaly_type = self.rng.choice(ANOMALY_TYPES)
            track_indices = df.index[mask]

            if anomaly_type == "route_deviation":
                df = self._inject_route_deviation(df, track_indices)
            elif anomaly_type == "speed_anomaly":
                df = self._inject_speed_anomaly(df, track_indices)
            elif anomaly_type == "loitering":
                df = self._inject_loitering(df, track_indices)
            elif anomaly_type == "dark_period":
                df = self._inject_dark_period(df, track_indices)

        return df

    def _inject_route_deviation(
        self, df: pd.DataFrame, indices: pd.Index
    ) -> pd.DataFrame:
        """Shift a segment of the track off-course."""
        n = len(indices)
        start = self.rng.integers(n // 4, n // 2)
        end = min(start + self.rng.integers(n // 6, n // 3), n)
        affected = indices[start:end]

        deviation_lat = self.rng.uniform(0.02, 0.08) * self.rng.choice([-1, 1])
        deviation_lon = self.rng.uniform(0.02, 0.08) * self.rng.choice([-1, 1])

        # Smooth deviation with a bell curve
        t = np.linspace(0, np.pi, len(affected))
        envelope = np.sin(t)

        df.loc[affected, "latitude"] += deviation_lat * envelope
        df.loc[affected, "longitude"] += deviation_lon * envelope
        df.loc[affected, "is_anomaly"] = True
        df.loc[affected, "anomaly_type"] = "route_deviation"
        return df

    def _inject_speed_anomaly(
        self, df: pd.DataFrame, indices: pd.Index
    ) -> pd.DataFrame:
        """Create sudden speed changes in a segment."""
        n = len(indices)
        start = self.rng.integers(n // 3, 2 * n // 3)
        end = min(start + self.rng.integers(5, max(6, n // 5)), n)
        affected = indices[start:end]

        multiplier = self.rng.choice([0.1, 0.2, 3.0, 5.0])
        df.loc[affected, "speed_knots"] *= multiplier
        df.loc[affected, "is_anomaly"] = True
        df.loc[affected, "anomaly_type"] = "speed_anomaly"
        return df

    def _inject_loitering(
        self, df: pd.DataFrame, indices: pd.Index
    ) -> pd.DataFrame:
        """Simulate circling/loitering at a point."""
        n = len(indices)
        center_idx = self.rng.integers(n // 3, 2 * n // 3)
        center_lat = df.loc[indices[center_idx], "latitude"]
        center_lon = df.loc[indices[center_idx], "longitude"]

        loiter_len = min(self.rng.integers(10, max(11, n // 4)), n - center_idx)
        affected = indices[center_idx : center_idx + loiter_len]

        # Create circular pattern
        angles = np.linspace(0, 2 * np.pi * self.rng.integers(1, 4), len(affected))
        radius = self.rng.uniform(0.005, 0.015)

        df.loc[affected, "latitude"] = center_lat + radius * np.cos(angles)
        df.loc[affected, "longitude"] = center_lon + radius * np.sin(angles)
        df.loc[affected, "speed_knots"] = self.rng.uniform(1.0, 3.0, len(affected))
        df.loc[affected, "is_anomaly"] = True
        df.loc[affected, "anomaly_type"] = "loitering"
        return df

    def _inject_dark_period(
        self, df: pd.DataFrame, indices: pd.Index
    ) -> pd.DataFrame:
        """Simulate a gap in reporting (AIS dark period).

        Marks the gap region and shifts subsequent positions to simulate
        the entity reappearing at an unexpected location.
        """
        n = len(indices)
        start = self.rng.integers(n // 4, n // 2)
        gap_len = min(self.rng.integers(5, max(6, n // 4)), n - start)
        affected = indices[start : start + gap_len]

        # Zero out speed during dark period and shift position
        offset_lat = self.rng.uniform(0.01, 0.04) * self.rng.choice([-1, 1])
        offset_lon = self.rng.uniform(0.01, 0.04) * self.rng.choice([-1, 1])

        df.loc[affected, "speed_knots"] = 0.0
        df.loc[affected, "latitude"] += offset_lat
        df.loc[affected, "longitude"] += offset_lon
        df.loc[affected, "is_anomaly"] = True
        df.loc[affected, "anomaly_type"] = "dark_period"
        return df
