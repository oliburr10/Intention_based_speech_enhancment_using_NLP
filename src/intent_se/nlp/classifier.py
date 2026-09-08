from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from intent_se.config import CLASS_ORDER, NLPConfig

__all__ = [
    "IntentClassifier",
    "CVResult",
    "build_classifiers",
    "cross_validate_all",
    "evaluate_on",
]


def build_classifiers(config: NLPConfig | None = None) -> dict[str, object]:
    """Construct the five candidate classifiers.
    """
    cfg = config or NLPConfig()

    models: dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=cfg.random_state,
        ),
        "LinearSVM": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=cfg.random_state,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=cfg.knn_neighbors,
            metric="cosine",
            weights="distance",
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=tuple(cfg.mlp_hidden),
            activation="relu",
            max_iter=800,
            early_stopping=True,
            random_state=cfg.random_state,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            random_state=cfg.random_state,
            verbosity=0,
        )
    except ImportError:  # pragma: no cover - optional dependency
        pass

    return models


@dataclass
class CVResult:
    """Cross-validation outcome for one classifier."""

    name: str
    mean_f1: float
    std_f1: float
    fold_scores: np.ndarray

    def __str__(self) -> str:
        return f"{self.name:<20s} macro-F1 {self.mean_f1:.3f} +/- {self.std_f1:.3f}"


def cross_validate_all(
    x: np.ndarray,
    y: np.ndarray,
    config: NLPConfig | None = None,
    models: dict[str, object] | None = None,
) -> list[CVResult]:
    """Run stratified k-fold CV for every candidate classifier.
    """
    cfg = config or NLPConfig()
    models = models if models is not None else build_classifiers(cfg)

    cv = StratifiedKFold(
        n_splits=cfg.cv_folds,
        shuffle=True,
        random_state=cfg.random_state,
    )

    results = []
    for name, model in models.items():
        scores = cross_val_score(model, x, y, cv=cv, scoring="f1_macro", n_jobs=1)
        results.append(
            CVResult(
                name=name,
                mean_f1=float(scores.mean()),
                std_f1=float(scores.std()),
                fold_scores=scores,
            )
        )

    return sorted(results, key=lambda r: r.mean_f1, reverse=True)


def evaluate_on(model: object, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-F1 of a fitted model on one split."""
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


class IntentClassifier:
    """Fitted intent classifier with label decoding and persistence.
    """

    def __init__(self, model: object | None = None) -> None:
        if model is None:
            model = build_classifiers()["LinearSVM"]
        self.model = model
        self.classes = list(CLASS_ORDER)
        self._fitted = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> IntentClassifier:
        """Fit on embeddings ``x`` and integer label ids ``y``."""
        self.model.fit(x, y)
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted. Call fit() or load() first.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict integer label ids."""
        self._check_fitted()
        return self.model.predict(x)

    def predict_labels(self, x: np.ndarray) -> list[str]:
        """Predict class names."""
        return [self.classes[i] for i in self.predict(x)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Class probabilities of shape ``(n_samples, 6)``.

        Raises
        ------
        AttributeError
            If the underlying estimator has no ``predict_proba``.
        """
        self._check_fitted()
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"{type(self.model).__name__} does not expose predict_proba."
            )
        return self.model.predict_proba(x)

    def confidence(self, x: np.ndarray) -> np.ndarray:
        """Highest class probability per sample, or 1.0 if unavailable.

        :class:`~intent_se.control.parameter_probe.ParameterProbe` gates on this
        so a low-confidence classification is ignored rather than acted on.

        .. note::
           Estimators without ``predict_proba`` fall back to 1.0 for every
           sample, which effectively disables that gate. All five candidates in
           :func:`build_classifiers` expose probabilities, but a substitution
           (``LinearSVC``, say) would silently remove the check.
        """
        try:
            return self.predict_proba(x).max(axis=1)
        except AttributeError:
            return np.ones(len(x))

    def save(self, path: str | Path) -> None:
        """Persist the fitted model with joblib."""
        import joblib

        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "classes": self.classes}, path)

    @classmethod
    def load(cls, path: str | Path) -> IntentClassifier:
        """Load a model saved by :meth:`save`."""
        import joblib

        payload = joblib.load(Path(path))
        obj = cls(payload["model"])
        obj.classes = payload["classes"]
        obj._fitted = True
        return obj


def cv_table(results: list[CVResult]) -> pd.DataFrame:
    """Format CV results as a table for reporting."""
    return pd.DataFrame(
        [{"classifier": r.name, "mean_macro_f1": round(r.mean_f1, 4),
          "std": round(r.std_f1, 4)} for r in results]
    ).set_index("classifier")
