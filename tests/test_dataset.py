"""Tests for dataset loading, validation and splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from intent_se.config import CLASS_ORDER, NLPConfig
from intent_se.nlp.dataset import class_distribution, load_dataset, split_dataset


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_dataset_has_expected_size(dataset):
    assert len(dataset) == 1106


def test_class_counts_match_the_documented_distribution(dataset):
    expected = {
        "TOO_NOISY": 225,
        "SPEECH_UNCLEAR": 194,
        "TOO_LOUD": 175,
        "TOO_QUIET": 170,
        "TOO_SHARP": 167,
        "BALANCED": 175,
    }
    assert dataset["label"].value_counts().to_dict() == expected


def test_no_duplicate_sentences(dataset):
    assert not dataset["sentence"].duplicated().any()


def test_severity_within_range(dataset):
    assert dataset["severity"].between(0.0, 1.0).all()


def test_balanced_class_is_always_zero_severity(dataset):
    assert (dataset.loc[dataset["label"] == "BALANCED", "severity"] == 0.0).all()


def test_every_complaint_class_has_nonzero_severity(dataset):
    complaints = dataset[dataset["label"] != "BALANCED"]
    assert (complaints["severity"] > 0.0).all()


def test_label_ids_follow_class_order(dataset):
    for idx, name in enumerate(CLASS_ORDER):
        assert (dataset.loc[dataset["label"] == name, "label_id"] == idx).all()


def test_split_is_stratified_and_disjoint(dataset):
    split = split_dataset(dataset, NLPConfig())

    total = len(split.train) + len(split.val) + len(split.test)
    assert total == len(dataset)

    # Development pool is train + val (940 sentences at the default fractions).
    assert len(split.dev) == len(split.train) + len(split.val)

    # No sentence appears in more than one split.
    sentences = pd.concat([split.train, split.val, split.test])["sentence"]
    assert not sentences.duplicated().any()

    # Every class is present in every split.
    for frame in (split.train, split.val, split.test):
        assert set(frame["label"]) == set(CLASS_ORDER)


def test_split_is_reproducible(dataset):
    a = split_dataset(dataset, NLPConfig(random_state=7))
    b = split_dataset(dataset, NLPConfig(random_state=7))
    pd.testing.assert_frame_equal(a.test, b.test)


def test_class_distribution_covers_all_classes(dataset):
    stats = class_distribution(dataset)
    assert list(stats.index) == list(CLASS_ORDER)
    assert stats["count"].sum() == len(dataset)


def test_rejects_unknown_labels(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("sentence,label,severity\nhello,NOT_A_CLASS,0.5\n")
    with pytest.raises(ValueError, match="Unknown labels"):
        load_dataset(bad)


def test_rejects_out_of_range_severity(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("sentence,label,severity\nhello,TOO_NOISY,1.5\n")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_dataset(bad)


def test_rejects_nonzero_balanced_severity(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("sentence,label,severity\nfine,BALANCED,0.4\n")
    with pytest.raises(ValueError, match="BALANCED"):
        load_dataset(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")
