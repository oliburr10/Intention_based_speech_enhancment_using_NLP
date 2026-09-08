"""
The classifier and the severity scorer run in parallel on the *same* embedding,
so a sentence is embedded once per query.
"""

from intent_se.nlp.classifier import IntentClassifier, build_classifiers, cross_validate_all
from intent_se.nlp.dataset import load_dataset, split_dataset
from intent_se.nlp.embeddings import SentenceEmbedder
from intent_se.nlp.severity import SeverityScorer

__all__ = [
    "IntentClassifier",
    "SentenceEmbedder",
    "SeverityScorer",
    "build_classifiers",
    "cross_validate_all",
    "load_dataset",
    "split_dataset",
]
