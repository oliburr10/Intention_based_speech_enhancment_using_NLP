"""End-to-end test of the training CLI.

The real pipeline embeds with ``all-mpnet-base-v2``, which is a 110M-parameter
download. To keep the suite fast, offline and deterministic, this test patches
:meth:`SentenceEmbedder.encode` with a TF-IDF + SVD projection.

That substitution tests the *plumbing* -- splitting, cross-validation, tie-
breaking, model persistence, table and figure generation -- not the quality of
the embeddings. Reported metrics from this test are meaningless as results; run
``python -m intent_se.cli.train_nlp`` with the real model for those.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

from intent_se.cli import train_nlp
from intent_se.nlp.embeddings import SentenceEmbedder


@pytest.fixture
def stub_embedder(monkeypatch):
    """Deterministic offline stand-in for the sentence transformer."""
    state: dict = {}

    def encode(self, sentences, **kwargs):  # noqa: ANN001
        sentences = list(sentences)
        if "model" not in state:
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=1),
                TruncatedSVD(n_components=64, random_state=0),
            )
            # Fit once on the full corpus so every call shares a projection.
            from intent_se.nlp.dataset import load_dataset

            model.fit(load_dataset()["sentence"].tolist())
            state["model"] = model

        vectors = state["model"].transform(sentences).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    monkeypatch.setattr(SentenceEmbedder, "encode", encode)
    monkeypatch.setattr(
        SentenceEmbedder, "encode_one", lambda self, s: encode(self, [s])[0]
    )
    return state


def test_training_cli_runs_end_to_end(stub_embedder, tmp_path, capsys):
    exit_code = train_nlp.main(
        ["--output-dir", str(tmp_path), "--no-figures", "--seed", "42"]
    )
    assert exit_code == 0

    # Models persisted.
    assert (tmp_path / "models" / "intent_classifier.joblib").exists()
    assert (tmp_path / "models" / "severity_scorer.joblib").exists()

    # Tables persisted.
    for name in (
        "cross_validation",
        "generalisation",
        "confusion_matrix",
        "severity_alpha",
        "severity_per_class",
    ):
        assert (tmp_path / "tables" / f"{name}.csv").exists(), name

    # Summary is well-formed.
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_sentences"] == 1106
    assert summary["selected_classifier"] in {
        "LogisticRegression", "LinearSVM", "kNN", "MLP", "XGBoost",
    }
    assert 0.0 <= summary["test_macro_f1"] <= 1.0
    assert summary["severity_alpha"] > 0

    out = capsys.readouterr().out
    assert "Selected:" in out
    assert "Confusion matrix" in out


def test_saved_models_are_usable_for_inference(stub_embedder, tmp_path):
    """The whole point of persisting the models: load them and classify."""
    from intent_se.control.mapping import ParameterController
    from intent_se.nlp.classifier import IntentClassifier
    from intent_se.nlp.severity import SeverityScorer

    train_nlp.main(["--output-dir", str(tmp_path), "--no-figures"])

    classifier = IntentClassifier.load(tmp_path / "models" / "intent_classifier.joblib")
    scorer = SeverityScorer.load(tmp_path / "models" / "severity_scorer.joblib")
    embedder = SentenceEmbedder()

    embedding = embedder.encode(["the background noise is unbearable"])
    intent = classifier.predict_labels(embedding)[0]
    severity = scorer.predict_one(embedding)

    assert intent in classifier.classes
    assert 0.0 <= severity <= 1.0

    # And the controller turns that into a parameter change.
    controller = ParameterController()
    update = controller.apply(intent, severity, float(classifier.confidence(embedding)[0]))
    assert isinstance(update.parameters, dict)
