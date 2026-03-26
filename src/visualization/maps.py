"""Interactive geospatial map visualization for trajectory data.

Creates folium maps showing vessel/aircraft/vehicle tracks with
color-coded anomaly highlighting and interactive popups.
"""

from __future__ import annotations

import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster, HeatMap
from dataclasses import dataclass

from src.data_generation.generator import POINTS_OF_INTEREST

# Color scheme for entity types and anomalies
ENTITY_COLORS = {
    "vessel": "#2196F3",
    "aircraft": "#4CAF50",
    "vehicle": "#FF9800",
}
ANOMALY_COLOR = "#F44336"
NORMAL_COLOR = "#888888"


@dataclass
class TrajectoryMapVisualizer:
    """Creates interactive folium maps for trajectory visualization.

    Attributes:
        center_lat: Map center latitude.
        center_lon: Map center longitude.
        zoom_start: Initial zoom level.
    """

    center_lat: float = 1.30
    center_lon: float = 103.90
    zoom_start: int = 10

    def create_track_map(
        self,
        df: pd.DataFrame,
        max_tracks: int = 50,
        show_anomalies: bool = True,
        show_pois: bool = True,
    ) -> folium.Map:
        """Create an interactive map showing trajectory tracks.

        Args:
            df: Trajectory DataFrame with anomaly labels.
            max_tracks: Maximum number of tracks to display.
            show_anomalies: Whether to highlight anomalous segments.
            show_pois: Whether to show points of interest.

        Returns:
            Folium Map object.
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles="CartoDB dark_matter",
        )

        track_ids = df["track_id"].unique()
        if len(track_ids) > max_tracks:
            rng = np.random.default_rng(42)
            # Ensure we include some anomalous tracks
            anomalous_ids = df[df["is_anomaly"]]["track_id"].unique()
            normal_ids = np.setdiff1d(track_ids, anomalous_ids)
            n_anom = min(len(anomalous_ids), max_tracks // 3)
            n_norm = max_tracks - n_anom
            selected = np.concatenate([
                rng.choice(anomalous_ids, size=n_anom, replace=False),
                rng.choice(normal_ids, size=min(n_norm, len(normal_ids)), replace=False),
            ])
            track_ids = selected

        # Draw tracks
        normal_group = folium.FeatureGroup(name="Normal Tracks")
        anomaly_group = folium.FeatureGroup(name="Anomalous Segments")

        for tid in track_ids:
            track = df[df["track_id"] == tid].sort_values("timestamp")
            entity_type = track["entity_type"].iloc[0]
            color = ENTITY_COLORS.get(entity_type, NORMAL_COLOR)
            has_anomaly = track["is_anomaly"].any()

            # Normal segments
            normal_points = track[~track["is_anomaly"]]
            if len(normal_points) > 1:
                coords = list(zip(normal_points["latitude"], normal_points["longitude"]))
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2,
                    opacity=0.6,
                    popup=f"Track {tid} ({entity_type})",
                ).add_to(normal_group)

            # Anomalous segments
            if show_anomalies and has_anomaly:
                anom_points = track[track["is_anomaly"]]
                if len(anom_points) > 0:
                    coords = list(zip(anom_points["latitude"], anom_points["longitude"]))
                    anomaly_type = anom_points["anomaly_type"].iloc[0]
                    folium.PolyLine(
                        coords,
                        color=ANOMALY_COLOR,
                        weight=4,
                        opacity=0.9,
                        popup=f"Track {tid} - {anomaly_type}",
                        dash_array="5 10",
                    ).add_to(anomaly_group)

                    # Mark anomaly start
                    folium.CircleMarker(
                        location=[anom_points.iloc[0]["latitude"],
                                  anom_points.iloc[0]["longitude"]],
                        radius=6,
                        color=ANOMALY_COLOR,
                        fill=True,
                        popup=f"Anomaly start: {anomaly_type}",
                    ).add_to(anomaly_group)

        normal_group.add_to(m)
        anomaly_group.add_to(m)

        # Points of interest
        if show_pois:
            poi_group = folium.FeatureGroup(name="Points of Interest")
            for name, (lat, lon) in POINTS_OF_INTEREST.items():
                folium.Marker(
                    location=[lat, lon],
                    popup=name.replace("_", " ").title(),
                    icon=folium.Icon(color="orange", icon="info-sign"),
                ).add_to(poi_group)
            poi_group.add_to(m)

        folium.LayerControl().add_to(m)
        return m

    def create_anomaly_heatmap(self, df: pd.DataFrame) -> folium.Map:
        """Create a heatmap of anomaly density.

        Args:
            df: Trajectory DataFrame with anomaly labels.

        Returns:
            Folium Map with heatmap overlay.
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles="CartoDB dark_matter",
        )

        anomalous = df[df["is_anomaly"]]
        if len(anomalous) > 0:
            heat_data = list(zip(
                anomalous["latitude"],
                anomalous["longitude"],
                anomalous["speed_knots"] / anomalous["speed_knots"].max(),
            ))
            HeatMap(
                heat_data,
                name="Anomaly Density",
                radius=15,
                blur=10,
                gradient={0.4: "blue", 0.65: "lime", 1: "red"},
            ).add_to(m)

        folium.LayerControl().add_to(m)
        return m

    def create_detection_map(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        model_name: str = "Model",
    ) -> folium.Map:
        """Create a map comparing ground truth anomalies with model detections.

        Args:
            df: Trajectory DataFrame with ground truth labels.
            predictions: Model predictions aligned with df rows.
            model_name: Name of the model for display.

        Returns:
            Folium Map showing true vs predicted anomalies.
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles="CartoDB positron",
        )

        df = df.copy()
        df["predicted"] = predictions

        # True positives, false positives, false negatives
        tp = df[(df["is_anomaly"]) & (df["predicted"] == 1)]
        fp = df[(~df["is_anomaly"]) & (df["predicted"] == 1)]
        fn = df[(df["is_anomaly"]) & (df["predicted"] == 0)]

        for label, data, color in [
            ("True Positive", tp, "green"),
            ("False Positive", fp, "orange"),
            ("False Negative", fn, "red"),
        ]:
            group = folium.FeatureGroup(name=f"{label} ({len(data)})")
            # Sample for performance
            sample = data.sample(min(200, len(data)), random_state=42) if len(data) > 200 else data
            for _, row in sample.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=3,
                    color=color,
                    fill=True,
                    opacity=0.7,
                    popup=f"{label}: Track {row['track_id']}",
                ).add_to(group)
            group.add_to(m)

        folium.LayerControl().add_to(m)
        return m
