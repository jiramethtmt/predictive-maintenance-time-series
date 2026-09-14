from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COUNTERS, HISTOGRAM_BINS, SPEC_COLUMNS, TIME_STEP, VEHICLE_ID, histogram_columns

WINDOW = 5


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.where(denominator > 0)).astype("float32")


def build_row_features(readouts: pd.DataFrame) -> pd.DataFrame:
    frame = readouts.sort_values([VEHICLE_ID, TIME_STEP], ignore_index=True)
    grouped = frame.groupby(VEHICLE_ID, sort=False)
    t = frame[TIME_STEP]

    readout_index = grouped.cumcount().astype("int32")
    elapsed = (t - grouped[TIME_STEP].transform("min")).astype("float32")
    columns: dict[str, pd.Series] = {
        VEHICLE_ID: frame[VEHICLE_ID],
        TIME_STEP: t,
        "readout_index": readout_index,
        "dt_last": grouped[TIME_STEP].diff().astype("float32"),
        "elapsed": elapsed,
        "dt_mean": _safe_divide(elapsed, readout_index),
    }

    window_dt = (t - grouped[TIME_STEP].shift(WINDOW)).astype("float32")
    for counter in COUNTERS:
        value = frame[counter]
        columns[f"{counter}_last"] = value.astype("float32")
        columns[f"{counter}_per_t"] = _safe_divide(value, t)
        columns[f"{counter}_rate_life"] = _safe_divide(
            value - grouped[counter].transform("first"), elapsed
        )
        columns[f"{counter}_rate_w"] = _safe_divide(
            value - grouped[counter].shift(WINDOW), window_dt
        )

    for family, bins in HISTOGRAM_BINS.items():
        cols = histogram_columns(family)
        block = frame[cols]
        volume = block.sum(axis=1)
        shares = block.div(volume.where(volume > 0), axis=0).astype("float32")
        columns[f"{family}_volume_log"] = np.log1p(volume).astype("float32")
        columns[f"{family}_reported"] = block[cols[0]].notna().astype("int8")
        # Bin shares are the "how it was used" signal; volume already lives in the counters.
        for i in range(bins):
            columns[f"{family}_s{i}"] = shares[cols[i]]
        columns[f"{family}_entropy"] = (
            -(shares * np.log(shares.where(shares > 0))).sum(axis=1, skipna=True).astype("float32")
        )

    return pd.concat(columns, axis=1)


def attach_specifications(features: pd.DataFrame, specifications: pd.DataFrame) -> pd.DataFrame:
    specs = specifications.copy()
    for column in SPEC_COLUMNS:
        specs[column] = specs[column].astype("category")
    return features.merge(specs, on=VEHICLE_ID, how="left")


def feature_matrix_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != VEHICLE_ID]
