"""Loading and cleaning the UCI Maternal Health Risk dataset.

Dataset: Ahmed, M. & Kashem, M. (2023). Maternal Health Risk. UCI Machine
Learning Repository. https://doi.org/10.24432/C5DP5D (CC BY 4.0)

1,014 records collected through an IoT-based risk monitoring system from
hospitals and community clinics in rural Bangladesh.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"
CLASSES = ["low risk", "mid risk", "high risk"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CSV = DATA_DIR / "maternal_health_risk.csv"
UCI_ID = 863


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Return the raw dataset, downloading it from UCI on first use."""
    path = Path(path) if path else DEFAULT_CSV
    if path.exists():
        return pd.read_csv(path)

    from ucimlrepo import fetch_ucirepo  # imported lazily: only needed once

    ds = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, normalise labels and drop exact duplicate rows.

    The raw file contains many exact duplicates. Left in, the same patient
    record can land in both the training and test folds, which inflates
    accuracy. We keep one copy of each.
    """
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    out = df[FEATURES + [TARGET]].copy()
    out[TARGET] = out[TARGET].str.strip().str.lower()
    unknown = set(out[TARGET]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected risk labels: {sorted(unknown)}")

    return out.drop_duplicates().reset_index(drop=True)


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    return clean(load_raw(path))
