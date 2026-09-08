"""Tests for the intent classifier and the model-selection helpers.

These use synthetic clustered vectors rather than real sentence embeddings, so
the suite runs without downloading the 110M-parameter transformer.
"""

from __future__ import annotations

import numpy as np
import pytest

from intent_se.config import CLASS_ORDER, NLPConfig
from intent_se.nlp.classifier import (
    IntentClassifier,
    build_classifiers,
    cross_validate_all,
    cv_table,
    evaluate_on,
)
from intent_se.nlp.evaluate import confusion_frame, generalisation_table, report


@pytest.fixture(scope="module")
def clustered():
    """Six well-separated Gaussian clusters, standing in for the six classes."""
    rng = np.random.default_rng(0)
    dim, per_class = 32, 60

    centers = rng.normal(size=(len(CLASS_ORDER), dim)) * 4.0
    x = np.concatenate([
        centers[i] + rng.normal(scale=0.7, size=(per_class, dim))
        for i in range(len(CLASS_ORDER))
    ])
    y = np.repeat(np.arange(len(CLASS_ORDER)), per_class)

    # L2-normalise, as the real embeddings are.
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), y


def test_build_classifiers_returns_the_candidate_set():
    models = build_classifiers()
    assert {"LogisticRegression", "LinearSVM", "kNN", "MLP"} <= set(models)


def test_knn_uses_cosine_distance():
    """Embeddings are L2-normalised, so cosine is the right metric."""
    knn = build_classifiers()["kNN"]
    assert knn.metric == "cosine"
    assert knn.n_neighbors == NLPConfig().knn_neighbors


def test_cross_validation_ranks_all_candidates(clustered):
    x, y = clustered
    results = cross_validate_all(x, y, NLPConfig())

    assert len(results) == len(build_classifiers())
    # Sorted best-first.
    assert results == sorted(results, key=lambda r: r.mean_f1, reverse=True)
    # Separable clusters should be classified well.
    assert results[0].mean_f1 > 0.9
    assert len(results[0].fold_scores) == NLPConfig().cv_folds


def test_cv_table_is_reportable(clustered):
    x, y = clustered
    table = cv_table(cross_validate_all(x, y, NLPConfig()))
    assert "mean_macro_f1" in table.columns
    assert table["mean_macro_f1"].is_monotonic_decreasing


def test_classifier_fit_and_predict(clustered):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)

    assert clf.predict(x).shape == y.shape
    assert evaluate_on(clf, x, y)["macro_f1"] > 0.9


def test_predict_labels_returns_class_names(clustered):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)
    labels = clf.predict_labels(x[:5])
    assert all(name in CLASS_ORDER for name in labels)


def test_confidence_is_a_probability(clustered):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)
    conf = clf.confidence(x[:20])
    assert conf.shape == (20,)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_confidence_falls_back_when_proba_unavailable(clustered):
    x, y = clustered
    clf = IntentClassifier(build_classifiers()["kNN"]).fit(x, y)
    assert clf.confidence(x[:5]).shape == (5,)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        IntentClassifier().predict(np.zeros((1, 32)))


def test_save_and_load_roundtrip(clustered, tmp_path):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)
    path = tmp_path / "clf.joblib"
    clf.save(path)

    loaded = IntentClassifier.load(path)
    np.testing.assert_array_equal(loaded.predict(x[:20]), clf.predict(x[:20]))
    assert loaded.classes == list(CLASS_ORDER)


def test_confusion_frame_is_square_and_labelled(clustered):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)
    cm = confusion_frame(y, clf.predict(x))

    assert cm.shape == (len(CLASS_ORDER), len(CLASS_ORDER))
    assert list(cm.index) == list(CLASS_ORDER)
    assert cm.to_numpy().sum() == len(y)


def test_report_names_every_class(clustered):
    x, y = clustered
    clf = IntentClassifier().fit(x, y)
    text = report(y, clf.predict(x))
    for name in CLASS_ORDER:
        assert name in text


def test_generalisation_table_sorts_by_drop():
    rows = [
        {"classifier": "A", "cv_f1": 0.96, "val_f1": 0.97, "test_f1": 0.93,
         "val_accuracy": 0.97, "test_accuracy": 0.93},
        {"classifier": "B", "cv_f1": 0.96, "val_f1": 0.96, "test_f1": 0.957,
         "val_accuracy": 0.96, "test_accuracy": 0.96},
    ]
    table = generalisation_table(rows)

    # B generalises better despite a lower validation score -- it must rank first.
    assert table.index[0] == "B"
    assert table.loc["B", "val_to_test_drop"] < table.loc["A", "val_to_test_drop"]
