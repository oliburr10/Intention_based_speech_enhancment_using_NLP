"""Continuous severity estimation.

Consider two users whose sentences both classify as ``TOO_NOISY``. One says
"there might be a barely noticeable background hum"; the other says "the
background noise today is absolutely excruciating". Same intent class,
completely different appropriate response. A system that applies a fixed
parameter change per class will either under-respond to severe complaints or
over-respond to mild ones.

The severity scorer runs *in parallel* with the intent classifier on the same
768-dim embedding and outputs a continuous score in ``[0, 1]``. The controller
uses it to scale the magnitude of the DSP adjustment.

.. note::
   This component was **not** in the original project proposal. It was added
   after it became clear that purely categorical intent mapping was too rigid
   for a system meant to respond proportionally to the user's discomfort.

Why Ridge rather than ordinary least squares
--------------------------------------------
With ~930 training samples and 768 embedding dimensions, ``N`` and ``D`` are
the same order of magnitude. OLS in that regime fits the training data closely
and generalises poorly. Ridge adds an L2 penalty that shrinks the weight vector
and forces the model onto the consistent patterns in the data rather than
memorising it.

Too little regularisation overfits; too much and the weights shrink so far that
the model predicts nearly the same severity regardless of the sentence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from intent_se.config import NLPConfig

__all__ = ["SeverityScorer", "AlphaSearchResult"]


@dataclass
class AlphaSearchResult:
    """Outcome of the regularisation-strength search."""

    best_alpha: float
    best_mae: float
    table: pd.DataFrame

    def __str__(self) -> str:
        return f"alpha={self.best_alpha:g} (CV MAE {self.best_mae:.4f})"


class SeverityScorer:
    """Ridge regression from sentence embedding to severity in ``[0, 1]``.

    Parameters
    ----------
    alpha:
        L2 regularisation strength. ``None`` means "choose by cross-validation
        during :meth:`fit`".
    config:
        Supplies the candidate alphas and fold count.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=(200, 16))
    >>> y = np.clip(x[:, 0] * 0.2 + 0.5, 0, 1)
    >>> scorer = SeverityScorer().fit(x, y)
    >>> float(scorer.predict(x[:1])[0]) >= 0.0
    True
    """

    def __init__(self, alpha: float | None = None, config: NLPConfig | None = None) -> None:
        self.cfg = config or NLPConfig()
        self.alpha = alpha
        self.model: Ridge | None = None
        self.search: AlphaSearchResult | None = None

    # ------------------------------------------------------------------

    def select_alpha(self, x: np.ndarray, y: np.ndarray) -> AlphaSearchResult:
        """Choose the regularisation strength by k-fold CV on MAE.

        MAE rather than MSE because severity is bounded in ``[0, 1]`` and an
        average absolute error is directly interpretable on that scale: an MAE
        of 0.10 means predictions are off by a tenth of the severity range.

        Parameters
        ----------
        x:
            Embeddings of shape ``(n_samples, dim)``.
        y:
            Severity targets in ``[0, 1]``.

        Returns
        -------
        AlphaSearchResult
        """
        cv = KFold(n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state)

        rows = []
        for alpha in self.cfg.ridge_alphas:
            scores = cross_val_score(
                Ridge(alpha=alpha),
                x,
                y,
                cv=cv,
                scoring="neg_mean_absolute_error",
            )
            rows.append({"alpha": alpha, "cv_mae": float(-scores.mean()),
                         "std": float(scores.std())})

        table = pd.DataFrame(rows).set_index("alpha").round(4)
        best_alpha = float(table["cv_mae"].idxmin())

        result = AlphaSearchResult(
            best_alpha=best_alpha,
            best_mae=float(table["cv_mae"].min()),
            table=table,
        )
        self.search = result
        return result

    def fit(self, x: np.ndarray, y: np.ndarray) -> SeverityScorer:
        """Fit the scorer, selecting ``alpha`` by CV if it was not given."""
        if self.alpha is None:
            self.alpha = self.select_alpha(x, y).best_alpha

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(x, y)
        return self

    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("Scorer is not fitted. Call fit() or load() first.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict severity, clipped to the valid ``[0, 1]`` range.

        Ridge is unconstrained and will occasionally predict slightly outside
        the range; clipping keeps the controller's arithmetic well-defined.
        """
        self._check_fitted()
        return np.clip(self.model.predict(x), 0.0, 1.0)

    def predict_one(self, x: np.ndarray) -> float:
        """Predict severity for a single embedding vector."""
        return float(self.predict(np.atleast_2d(x))[0])

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """MAE and R^2 on a held-out split."""
        pred = self.predict(x)
        return {
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }

    def evaluate_per_class(
        self, x: np.ndarray, y: np.ndarray, labels: list[str]
    ) -> pd.DataFrame:
        """Per-class MAE, which is where the interesting failures show up.

        ``TOO_QUIET`` is expected to be the weakest class: its sentences are too
        semantically similar to each other across different severity levels, so
        the embedding gives the regressor little to separate mild from severe.
        """
        pred = self.predict(x)
        frame = pd.DataFrame({"label": labels, "true": y, "pred": pred})
        frame["abs_error"] = (frame["true"] - frame["pred"]).abs()
        return (
            frame.groupby("label")["abs_error"]
            .agg(["count", "mean", "max"])
            .rename(columns={"mean": "mae", "max": "worst"})
            .round(4)
            .sort_values("mae")
        )

    def save(self, path: str | Path) -> None:
        """Persist the fitted scorer."""
        import joblib

        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "alpha": self.alpha}, path)

    @classmethod
    def load(cls, path: str | Path) -> SeverityScorer:
        """Load a scorer saved by :meth:`save`."""
        import joblib

        payload = joblib.load(Path(path))
        obj = cls(alpha=payload["alpha"])
        obj.model = payload["model"]
        return obj
