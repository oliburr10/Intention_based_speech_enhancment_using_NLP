"""Tests for the Ridge severity scorer."""

from __future__ import annotations

import numpy as np
import pytest

from intent_se.config import NLPConfig
from intent_se.nlp.severity import SeverityScorer


@pytest.fixture
def synthetic():
    """A learnable severity signal in a high-dimensional embedding-like space."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 64))
    weights = rng.normal(size=64) * 0.1
    # Label noise, so the target is learnable but not perfectly recoverable --
    # a noiseless target makes every alpha score an identical MAE of 0.
    y = np.clip(0.5 + x @ weights * 0.1 + rng.normal(scale=0.05, size=400), 0.0, 1.0)
    return x, y


def test_alpha_selection_picks_a_candidate(synthetic):
    x, y = synthetic
    scorer = SeverityScorer(config=NLPConfig())
    result = scorer.select_alpha(x, y)

    assert result.best_alpha in NLPConfig().ridge_alphas
    assert result.best_mae > 0
    assert len(result.table) == len(NLPConfig().ridge_alphas)


def test_fit_without_alpha_selects_by_cv(synthetic):
    x, y = synthetic
    scorer = SeverityScorer().fit(x, y)
    assert scorer.alpha is not None
    assert scorer.search is not None


def test_explicit_alpha_skips_the_search(synthetic):
    x, y = synthetic
    scorer = SeverityScorer(alpha=0.10).fit(x, y)
    assert scorer.alpha == 0.10
    assert scorer.search is None


def test_predictions_are_clipped_to_unit_range():
    """Ridge is unconstrained, so clipping keeps the controller well-defined."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 8))
    y = np.clip(x[:, 0], 0, 1)

    scorer = SeverityScorer(alpha=1e-6).fit(x, y)
    pred = scorer.predict(rng.normal(size=(50, 8)) * 20)

    assert pred.min() >= 0.0
    assert pred.max() <= 1.0


def test_predict_one_returns_a_scalar(synthetic):
    x, y = synthetic
    scorer = SeverityScorer(alpha=0.1).fit(x, y)
    value = scorer.predict_one(x[0])
    assert isinstance(value, float)
    assert 0.0 <= value <= 1.0


def test_evaluate_reports_mae_and_r2(synthetic):
    x, y = synthetic
    scorer = SeverityScorer(alpha=0.1).fit(x, y)
    metrics = scorer.evaluate(x, y)
    assert set(metrics) == {"mae", "r2"}
    assert metrics["mae"] < 0.5


def test_per_class_evaluation_groups_correctly(synthetic):
    x, y = synthetic
    labels = ["TOO_NOISY"] * 200 + ["TOO_QUIET"] * 200
    scorer = SeverityScorer(alpha=0.1).fit(x, y)
    table = scorer.evaluate_per_class(x, y, labels)

    assert set(table.index) == {"TOO_NOISY", "TOO_QUIET"}
    assert table["count"].sum() == 400


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        SeverityScorer().predict(np.zeros((1, 8)))


def test_save_and_load_roundtrip(synthetic, tmp_path):
    x, y = synthetic
    scorer = SeverityScorer(alpha=0.1).fit(x, y)
    path = tmp_path / "scorer.joblib"
    scorer.save(path)

    loaded = SeverityScorer.load(path)
    np.testing.assert_allclose(loaded.predict(x[:10]), scorer.predict(x[:10]))
    assert loaded.alpha == 0.1
