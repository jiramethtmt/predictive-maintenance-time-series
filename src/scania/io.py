from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import TIME_STEP, VEHICLE_ID, feature_columns

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

LABEL_FILE = {"train": "train_tte.csv", "validation": "validation_labels.csv", "test": "test_labels.csv"}


@dataclass(frozen=True)
class Split:
    name: str
    readouts: pd.DataFrame
    labels: pd.DataFrame
    specifications: pd.DataFrame


def _readout_dtypes() -> dict[str, str]:
    # float32 halves the 935k x 105 train frame; counter magnitudes stay well inside float32 range.
    return {VEHICLE_ID: "int32", TIME_STEP: "float32"} | {c: "float32" for c in feature_columns()}


def load_readouts(split: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / f"{split}_operational_readouts.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    frame = pd.read_csv(RAW_DIR / f"{split}_operational_readouts.csv", dtype=_readout_dtypes())
    frame = frame.sort_values([VEHICLE_ID, TIME_STEP], ignore_index=True)
    frame.to_parquet(cached, index=False)
    return frame


def load_split(name: str) -> Split:
    return Split(
        name=name,
        readouts=load_readouts(name),
        labels=pd.read_csv(RAW_DIR / LABEL_FILE[name]),
        specifications=pd.read_csv(RAW_DIR / f"{name}_specifications.csv"),
    )
