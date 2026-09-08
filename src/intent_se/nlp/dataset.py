


"""Loading and splitting the hearing-aid complaint dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from intent_se.config import CLASS_ORDER, NLPConfig

__all__ = ["DatasetSplit", "load_dataset", "split_dataset", "class_distribution"]

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "complaints_v4.csv"


@dataclass
class DatasetSplit:
    """A stratified train/validation/test partition.

    Cross-validation runs on ``train + val`` (the *development pool*); ``test``
    is held back and used exactly once, after model selection is complete.
    That single use is what makes the reported test score an unbiased estimate.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def dev(self) -> pd.DataFrame:
        """Train + validation combined -- the pool cross-validation runs on."""
        return pd.concat([self.train, self.val], ignore_index=True)

    def summary(self) -> pd.DataFrame:
        """Per-split sample counts by class."""
        rows = []
        for name, frame in (("train", self.train), ("val", self.val), ("test", self.test)):
            counts = frame["label"].value_counts()
            rows.append({"split": name, "total": len(frame), **counts.to_dict()})
        return pd.DataFrame(rows).fillna(0).set_index("split")


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the complaint dataset from CSV
    """
    path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)

    missing = {"sentence", "label", "severity"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    unknown = set(df["label"]) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unknown labels in dataset: {sorted(unknown)}")

    if not df["severity"].between(0.0, 1.0).all():
        raise ValueError("Severity values must lie in [0, 1].")

    neutral = df[df["label"] == "BALANCED"]["severity"]
    if not (neutral == 0.0).all():
        raise ValueError("Every BALANCED sentence must have severity exactly 0.0.")

    df["label_id"] = df["label"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    return df


def split_dataset(df: pd.DataFrame, config: NLPConfig | None = None) -> DatasetSplit:
    """Split into stratified train / validation / test sets
    """
    cfg = config or NLPConfig()

    holdout = cfg.val_size + cfg.test_size
    train, rest = train_test_split(
        df,
        test_size=holdout,
        stratify=df["label"],
        random_state=cfg.random_state,
    )
    # Split the holdout evenly into validation and test.
    val, test = train_test_split(
        rest,
        test_size=cfg.test_size / holdout,
        stratify=rest["label"],
        random_state=cfg.random_state,
    )

    return DatasetSplit(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class counts and severity statistics.

    Returns
    -------
    pd.DataFrame
        Indexed by class in :data:`intent_se.config.CLASS_ORDER` order, with
        ``count``, ``mean``, ``min`` and ``max`` severity.
    """
    stats = df.groupby("label")["severity"].agg(["count", "mean", "min", "max"])
    return stats.reindex(list(CLASS_ORDER)).round(3)
