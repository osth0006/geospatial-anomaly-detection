"""Synthetic geospatial trajectory data generator.

Generates realistic vessel, aircraft, and vehicle movement tracks with
configurable parameters for route patterns, speed profiles, and noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RouteConfig:
    """Configuration for a single route template."""

    name: str
    waypoints: list[tuple[float, float]]  # (lat, lon) pairs
    base_speed_knots: float = 12.0
    speed_variance: float = 2.0
    heading_noise_deg: float = 5.0
    position_noise_km: float = 0.5
    entity_type: Literal["vessel", "aircraft", "vehicle"] = "vessel"


# Pre-defined route templates simulating realistic maritime corridors
DEFAULT_ROUTES = [
    RouteConfig(
        name="strait_transit_east",
        waypoints=[
            (1.15, 103.60), (1.20, 103.80), (1.25, 104.00),
            (1.30, 104.20), (1.28, 104.40),
        ],
        base_speed_knots=10.0,
        entity_type="vessel",
    ),
    RouteConfig(
        name="strait_transit_west",
        waypoints=[
            (1.28, 104.40), (1.30, 104.20), (1.25, 104.00),
            (1.20, 103.80), (1.15, 103.60),
        ],
        base_speed_knots=11.0,
        entity_type="vessel",
    ),
    RouteConfig(
        name="coastal_patrol",
        waypoints=[
            (1.30, 103.70), (1.35, 103.75), (1.40, 103.80),
            (1.38, 103.90), (1.33, 103.85), (1.30, 103.70),
        ],
        base_speed_knots=15.0,
        speed_variance=3.0,
        entity_type="vessel",
    ),
    RouteConfig(
        name="air_corridor_north",
        waypoints=[
            (1.10, 103.90), (1.50, 103.95), (2.00, 104.00),
            (2.50, 104.10), (3.00, 104.20),
        ],
        base_speed_knots=250.0,
        speed_variance=15.0,
        heading_noise_deg=2.0,
        position_noise_km=0.2,
        entity_type="aircraft",
    ),
    RouteConfig(
        name="vehicle_route_a",
        waypoints=[
            (1.30, 103.80), (1.31, 103.82), (1.32, 103.85),
            (1.34, 103.87), (1.35, 103.90),
        ],
        base_speed_knots=20.0,
        speed_variance=5.0,
        heading_noise_deg=8.0,
        position_noise_km=0.05,
        entity_type="vehicle",
    ),
]

# Points of interest for proximity feature computation
POINTS_OF_INTEREST = {
    "port_alpha": (1.26, 103.82),
    "port_bravo": (1.29, 104.10),
    "anchorage_zone": (1.22, 103.95),
    "restricted_area": (1.35, 104.05),
    "airfield": (1.50, 103.95),
}


@dataclass
class TrajectoryGenerator:
    """Generates synthetic geospatial trajectory datasets.

    Creates realistic movement tracks by interpolating between waypoints
    with configurable noise, speed variation, and temporal patterns.

    Attributes:
        routes: Route templates to sample from.
        time_step_seconds: Interval between position reports.
        seed: Random seed for reproducibility.
    """

    routes: list[RouteConfig] = field(default_factory=lambda: DEFAULT_ROUTES)
    time_step_seconds: int = 60
    seed: int = 42

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def generate_dataset(
        self,
        n_tracks: int = 200,
        start_time: str = "2025-01-01",
        anomaly_ratio: float = 0.0,
    ) -> pd.DataFrame:
        """Generate a dataset of trajectory tracks.

        Args:
            n_tracks: Number of tracks to generate.
            start_time: Base start time for tracks.
            anomaly_ratio: Unused here; anomalies injected separately.

        Returns:
            DataFrame with columns: track_id, timestamp, latitude, longitude,
            speed_knots, heading_deg, entity_type, route_name.
        """
        all_tracks = []
        base_time = pd.Timestamp(start_time)

        for i in range(n_tracks):
            route = self.routes[self.rng.integers(0, len(self.routes))]
            track_start = base_time + pd.Timedelta(
                hours=float(self.rng.uniform(0, 720))
            )
            track_df = self._generate_single_track(i, route, track_start)
            all_tracks.append(track_df)

        df = pd.concat(all_tracks, ignore_index=True)
        df = df.sort_values(["track_id", "timestamp"]).reset_index(drop=True)
        return df

    def _generate_single_track(
        self, track_id: int, route: RouteConfig, start_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Generate a single trajectory by interpolating route waypoints."""
        waypoints = np.array(route.waypoints)
        n_waypoints = len(waypoints)

        # Compute segment distances and allocate points proportionally
        segment_dists = []
        for j in range(n_waypoints - 1):
            d = self._haversine_km(waypoints[j], waypoints[j + 1])
            segment_dists.append(d)
        total_dist = sum(segment_dists)

        # Speed in km per time step
        speed_kmh = route.base_speed_knots * 1.852
        dist_per_step = speed_kmh * (self.time_step_seconds / 3600)

        total_steps = max(int(total_dist / dist_per_step), 10)

        # Interpolate positions along the route
        positions = self._interpolate_route(waypoints, total_steps)

        # Add noise
        lat_noise = self.rng.normal(0, route.position_noise_km / 111.0, total_steps)
        lon_noise = self.rng.normal(0, route.position_noise_km / 111.0, total_steps)
        positions[:, 0] += lat_noise
        positions[:, 1] += lon_noise

        # Compute derived fields
        timestamps = [
            start_time + pd.Timedelta(seconds=self.time_step_seconds * s)
            for s in range(total_steps)
        ]

        speeds = np.clip(
            route.base_speed_knots + self.rng.normal(0, route.speed_variance, total_steps),
            0.5,
            route.base_speed_knots * 3,
        )

        headings = self._compute_headings(positions)
        headings += self.rng.normal(0, route.heading_noise_deg, total_steps)
        headings = headings % 360

        return pd.DataFrame({
            "track_id": track_id,
            "timestamp": timestamps,
            "latitude": positions[:, 0],
            "longitude": positions[:, 1],
            "speed_knots": speeds,
            "heading_deg": headings,
            "entity_type": route.entity_type,
            "route_name": route.name,
        })

    def _interpolate_route(
        self, waypoints: np.ndarray, n_points: int
    ) -> np.ndarray:
        """Linearly interpolate positions along waypoint segments."""
        cumulative = [0.0]
        for i in range(len(waypoints) - 1):
            d = self._haversine_km(waypoints[i], waypoints[i + 1])
            cumulative.append(cumulative[-1] + d)
        total = cumulative[-1]

        target_dists = np.linspace(0, total, n_points)
        positions = np.zeros((n_points, 2))

        seg_idx = 0
        for i, td in enumerate(target_dists):
            while seg_idx < len(waypoints) - 2 and td > cumulative[seg_idx + 1]:
                seg_idx += 1
            seg_start = cumulative[seg_idx]
            seg_end = cumulative[seg_idx + 1]
            if seg_end - seg_start < 1e-9:
                frac = 0.0
            else:
                frac = (td - seg_start) / (seg_end - seg_start)
            positions[i] = (
                waypoints[seg_idx] + frac * (waypoints[seg_idx + 1] - waypoints[seg_idx])
            )

        return positions

    @staticmethod
    def _haversine_km(p1: np.ndarray, p2: np.ndarray) -> float:
        """Haversine distance between two (lat, lon) points in km."""
        lat1, lon1 = np.radians(p1)
        lat2, lon2 = np.radians(p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371 * 2 * np.arcsin(np.sqrt(a))

    @staticmethod
    def _compute_headings(positions: np.ndarray) -> np.ndarray:
        """Compute bearing between consecutive positions."""
        n = len(positions)
        headings = np.zeros(n)
        for i in range(n - 1):
            lat1, lon1 = np.radians(positions[i])
            lat2, lon2 = np.radians(positions[i + 1])
            dlon = lon2 - lon1
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            bearing = np.degrees(np.arctan2(x, y))
            headings[i] = bearing % 360
        headings[-1] = headings[-2] if n > 1 else 0
        return headings
