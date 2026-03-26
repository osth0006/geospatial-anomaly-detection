"""Evaluation visualization plots for anomaly detection models.

Generates ROC curves, precision-recall curves, and model comparison
charts using matplotlib and plotly.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix,
)
from dataclasses import dataclass


@dataclass
class EvaluationPlotter:
    """Creates evaluation plots for anomaly detection results.

    Attributes:
        figsize: Default matplotlib figure size.
        style: Matplotlib style to use.
    """

    figsize: tuple[int, int] = (10, 6)
    style: str = "seaborn-v0_8-darkgrid"

    def plot_roc_curves(
        self,
        results: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str = "ROC Curves - Anomaly Detection Models",
    ) -> plt.Figure:
        """Plot ROC curves for multiple models.

        Args:
            results: Dict mapping model name -> (y_true, y_scores).
            title: Plot title.

        Returns:
            Matplotlib Figure.
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
        for i, (name, (y_true, y_scores)) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})",
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        return fig

    def plot_precision_recall(
        self,
        results: dict[str, tuple[np.ndarray, np.ndarray]],
        title: str = "Precision-Recall Curves",
    ) -> plt.Figure:
        """Plot precision-recall curves for multiple models.

        Args:
            results: Dict mapping model name -> (y_true, y_scores).
            title: Plot title.

        Returns:
            Matplotlib Figure.
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
        for i, (name, (y_true, y_scores)) in enumerate(results.items()):
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            ax.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linewidth=2,
                label=f"{name} (AP = {ap:.3f})",
            )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        return fig

    def plot_confusion_matrices(
        self,
        results: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> plt.Figure:
        """Plot confusion matrices for multiple models side by side.

        Args:
            results: Dict mapping model name -> (y_true, y_pred).

        Returns:
            Matplotlib Figure.
        """
        plt.style.use(self.style)
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1:
            axes = [axes]

        for ax, (name, (y_true, y_pred)) in zip(axes, results.items()):
            cm = confusion_matrix(y_true, y_pred)
            im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Normal", "Anomaly"])
            ax.set_yticklabels(["Normal", "Anomaly"])

            # Annotate cells
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center",
                            fontsize=14, fontweight="bold",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.tight_layout()
        return fig

    def plot_interactive_comparison(
        self,
        results: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> go.Figure:
        """Create interactive plotly figure comparing model performance.

        Args:
            results: Dict mapping model name -> (y_true, y_scores).

        Returns:
            Plotly Figure with ROC and PR subplots.
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ROC Curves", "Precision-Recall Curves"),
        )

        colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800"]
        for i, (name, (y_true, y_scores)) in enumerate(results.items()):
            color = colors[i % len(colors)]

            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    name=f"{name} (AUC={roc_auc:.3f})",
                    line=dict(color=color, width=2),
                    legendgroup=name,
                ),
                row=1, col=1,
            )

            # PR
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision, mode="lines",
                    name=f"{name} (AP={ap:.3f})",
                    line=dict(color=color, width=2, dash="dot"),
                    legendgroup=name,
                    showlegend=False,
                ),
                row=1, col=2,
            )

        # Random baseline on ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Random",
                line=dict(color="gray", dash="dash"),
            ),
            row=1, col=1,
        )

        fig.update_layout(
            title="Anomaly Detection Model Comparison",
            height=450,
            width=1000,
            template="plotly_dark",
        )
        fig.update_xaxes(title_text="FPR", row=1, col=1)
        fig.update_yaxes(title_text="TPR", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)

        return fig
