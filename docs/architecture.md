# Architecture

## System Design

The geospatial anomaly detection system follows a modular pipeline architecture:

```
Raw Data → Feature Engineering → Model Training → Evaluation → Visualization
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `data_generation` | Synthetic trajectory generation with configurable routes and anomaly injection |
| `feature_engineering` | Spatial/temporal/kinematic feature extraction at point and track levels |
| `models` | Three detection approaches: Autoencoder, Isolation Forest, LSTM |
| `evaluation` | Metrics computation and multi-model comparison |
| `visualization` | Interactive maps (folium) and evaluation plots (matplotlib/plotly) |

### Data Flow

1. **TrajectoryGenerator** produces position reports along predefined routes with realistic noise
2. **AnomalyInjector** modifies a fraction of tracks with anomalous behaviors
3. **TrajectoryFeatureExtractor** computes derived features at point and track granularity
4. Models consume either track-level feature vectors (Autoencoder, Isolation Forest) or sliding-window sequences (LSTM)
5. **AnomalyEvaluator** computes metrics; visualizers render results

### Anomaly Types

| Type | Description | Detection Signal |
|------|-------------|-----------------|
| Route Deviation | Off-course movement with bell-curve displacement | High reconstruction error, spatial outlier |
| Speed Anomaly | Sudden acceleration/deceleration | Speed z-score spike, kinematic feature outlier |
| Loitering | Circular pattern at unexpected location | Low speed, high heading change, small bounding box |
| Dark Period | Gap in reporting with position shift | Zero speed, position discontinuity |
