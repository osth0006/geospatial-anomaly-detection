"""Evaluation metrics for anomaly detection models.

Computes standard classification metrics and provides comparison
functionality across multiple detection approaches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from dataclasses import dataclass


@dataclass
class AnomalyEvaluator:
    """Evaluates and compares anomaly detection models.

    Computes standard metrics and generates comparison tables.
    """

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray | None = None,
        model_name: str = "Model",
    ) -> dict:
        """Compute evaluation metrics for a single model.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            y_scores: Continuous anomaly scores (for AUC computation).
            model_name: Name identifier for the model.

        Returns:
            Dictionary of metric name -> value.
        """
        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_scores is not None and len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
            metrics["avg_precision"] = average_precision_score(y_true, y_scores)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
            metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return metrics

    def compare_models(
        self,
        results: dict[str, dict],
    ) -> pd.DataFrame:
        """Create a comparison table across multiple models.

        Args:
            results: Dict mapping model name -> metrics dict.

        Returns:
            DataFrame with one row per model and metrics as columns.
        """
        rows = list(results.values())
        comparison = pd.DataFrame(rows)
        comparison = comparison.set_index("model")

        # Format numeric columns
        numeric_cols = comparison.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ("true_positives", "false_positives",
                           "true_negatives", "false_negatives"):
                comparison[col] = comparison[col].round(4)

        return comparison

    def print_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ) -> str:
        """Generate a formatted classification report string.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels.
            model_name: Name identifier for the model.

        Returns:
            Formatted report string.
        """
        header = f"\n{'='*60}\n{model_name} Classification Report\n{'='*60}\n"
        report = classification_report(
            y_true, y_pred,
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        )
        return header + report
