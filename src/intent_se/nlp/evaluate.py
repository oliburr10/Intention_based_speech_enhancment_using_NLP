"""Evaluation and figure generation for the NLP pipeline.

Produces the four result artefacts reported in the thesis:

1. The cross-validation table across the five candidate classifiers.
2. The validation-to-test generalisation table that breaks the CV tie.
3. The confusion matrix of the selected classifier.
4. The t-SNE projection of the embedding space.

Plus the severity scorer's alpha-selection table and per-class error scatter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix

from intent_se.config import CLASS_ORDER, NLPConfig

__all__ = [
    "confusion_frame",
    "generalisation_table",
    "plot_confusion_matrix",
    "plot_severity_scatter",
    "plot_tsne",
    "report",
]


def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Per-class precision/recall/F1 as a formatted string."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_ORDER))),
        target_names=list(CLASS_ORDER),
        digits=3,
        zero_division=0,
    )


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    return pd.DataFrame(matrix, index=list(CLASS_ORDER), columns=list(CLASS_ORDER))


def generalisation_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["val_to_test_drop"] = (frame["val_f1"] - frame["test_f1"]).round(4)
    return frame.set_index("classifier").sort_values("val_to_test_drop").round(4)

# Figures

def _savefig(fig, out_path: str | Path | None):
    """Save to ``out_path`` if given; always return the figure."""
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str | Path | None = None,
    normalize: bool = False,
):
    """Render the confusion matrix.

    Accuracy and F1 say how well the classifier does overall; only the
    confusion matrix says *where* it fails.
    """
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_ORDER))))
    display = matrix.astype(float)
    if normalize:
        display = display / np.maximum(display.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(display, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Intent classification -- confusion matrix (test set)")

    threshold = display.max() / 2.0
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if display[i, j] > threshold else "black",
                fontsize=10,
            )

    fig.tight_layout()
    return _savefig(fig, out_path)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path | None = None,
    config: NLPConfig | None = None,
    perplexity: float = 30.0,
):

    import matplotlib.pyplot as plt

    cfg = config or NLPConfig()

    projection = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5.0, (len(embeddings) - 1) / 3)),
        init="pca",
        learning_rate="auto",
        random_state=cfg.random_state,
    ).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = plt.get_cmap("tab10")

    for idx, name in enumerate(CLASS_ORDER):
        mask = labels == idx
        if not mask.any():
            continue
        ax.scatter(
            projection[mask, 0], projection[mask, 1],
            s=22, alpha=0.75, color=colors(idx), label=name,
            edgecolors="none",
        )

    ax.set_title("t-SNE projection of sentence embeddings")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return _savefig(fig, out_path)


def plot_severity_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: str | Path | None = None,
):
    """Predicted vs. labelled severity, one panel per intent class."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    labels = np.asarray(labels)

    for ax, name in zip(axes.ravel(), CLASS_ORDER, strict=True):
        mask = labels == name
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        if mask.any():
            ax.scatter(y_true[mask], y_pred[mask], s=20, alpha=0.7)
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            ax.set_title(f"{name}  (MAE {mae:.3f})", fontsize=10)
        else:
            ax.set_title(name, fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)

    fig.supxlabel("Labelled severity")
    fig.supylabel("Predicted severity")
    fig.suptitle("Severity scorer -- predicted vs. labelled, by class")
    fig.tight_layout()
    return _savefig(fig, out_path)
