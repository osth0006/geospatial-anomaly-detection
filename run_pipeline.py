"""Run the full geospatial anomaly detection pipeline.

Generates synthetic trajectory data, injects anomalies, extracts features,
trains models, and prints evaluation results.
"""

import numpy as np
from src.data_generation.generator import TrajectoryGenerator
from src.data_generation.anomalies import AnomalyInjector
from src.feature_engineering.features import TrajectoryFeatureExtractor
from src.models.autoencoder import AutoencoderDetector
from src.models.isolation_forest import IsolationForestDetector
from src.evaluation.metrics import AnomalyEvaluator


def main():
    print("=" * 60)
    print("Geospatial Anomaly Detection Pipeline")
    print("=" * 60)

    # Generate synthetic trajectories
    print("\n[1/5] Generating synthetic trajectories...")
    generator = TrajectoryGenerator()
    df = generator.generate_dataset(n_tracks=300)
    n_tracks = df["track_id"].nunique()
    print(f"  Generated {len(df)} points across {n_tracks} tracks")

    # Inject anomalies
    print("\n[2/5] Injecting anomalies...")
    injector = AnomalyInjector(anomaly_fraction=0.15)
    df = injector.inject(df)
    n_anomalous = df.groupby("track_id")["is_anomaly"].any().sum()
    print(f"  {n_anomalous} tracks contain anomalies")

    # Extract features
    print("\n[3/5] Extracting features...")
    extractor = TrajectoryFeatureExtractor()
    df_points = extractor.extract_point_features(df)
    track_features = extractor.extract_track_features(df_points)
    print(f"  Track-level features: {track_features.shape}")

    # Prepare feature matrix and labels
    label_col = "is_anomalous"
    feature_cols = [
        c for c in track_features.columns
        if c not in ("track_id", "entity_type", label_col)
    ]
    X = track_features[feature_cols].values.astype(np.float32)
    y = track_features[label_col].values.astype(int)

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Train models
    print("\n[4/5] Training models...")

    print("  Training Isolation Forest...")
    iso = IsolationForestDetector()
    iso.fit(X)
    iso_pred = iso.predict(X)

    print("  Training Autoencoder...")
    ae = AutoencoderDetector(epochs=50)
    ae.fit(X)
    ae_pred = ae.predict(X)

    # Evaluate
    print("\n[5/5] Evaluating models...")
    evaluator = AnomalyEvaluator()

    results = {}
    for name, preds in [("IsolationForest", iso_pred), ("Autoencoder", ae_pred)]:
        metrics = evaluator.evaluate(y, preds, model_name=name)
        results[name] = metrics
        print(evaluator.print_report(y, preds, model_name=name))

    comparison = evaluator.compare_models(results)
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(comparison.to_string())

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
