from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import COUNTERS, HISTOGRAM_BINS, SPEC_COLUMNS, TIME_STEP, VEHICLE_ID, histogram_columns

WINDOWS: tuple[int, ...] = (3, 10, 20)
ACCELERATION_WINDOW = 3


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.where(denominator > 0)).astype("float32")


def build_row_features(readouts: pd.DataFrame) -> pd.DataFrame:
    frame = readouts.sort_values([VEHICLE_ID, TIME_STEP], ignore_index=True)
    # Missing readouts arrive a whole histogram family at a time, which is a sensor that stopped
    # reporting rather than a stray null. Every measured column is cumulative, so carrying the last
    # reported value forward is the faithful fill; the `_reported` flags below still carry the fact
    # that it was absent. Forward only: a backward fill would let a cut point see readouts that
    # come after it, which inference never can because the series is truncated there.
    measured = list(COUNTERS) + [c for f in HISTOGRAM_BINS for c in histogram_columns(f)]
    reported = frame[measured].notna()
    frame[measured] = frame.groupby(VEHICLE_ID, sort=False)[measured].ffill()
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

    window_dt = {w: (t - grouped[TIME_STEP].shift(w)).astype("float32") for w in WINDOWS}
    for counter in COUNTERS:
        value = frame[counter]
        life_rate = _safe_divide(value - grouped[counter].transform("first"), elapsed)
        columns[f"{counter}_last"] = value.astype("float32")
        columns[f"{counter}_per_t"] = _safe_divide(value, t)
        columns[f"{counter}_rate_life"] = life_rate
        for w in WINDOWS:
            rate = _safe_divide(value - grouped[counter].shift(w), window_dt[w])
            columns[f"{counter}_rate_w{w}"] = rate
            if w == ACCELERATION_WINDOW:
                # Recent wear against the truck's own lifetime average: above 1 means this truck
                # just started working harder than it ever has, which is the degradation signal
                # the raw level cannot express.
                columns[f"{counter}_accel"] = _safe_divide(rate, life_rate)

    for family, bins in HISTOGRAM_BINS.items():
        cols = histogram_columns(family)
        block = frame[cols]
        volume = block.sum(axis=1)
        shares = block.div(volume.where(volume > 0), axis=0).astype("float32")
        baseline = shares.groupby(frame[VEHICLE_ID], sort=False).transform("first")
        entropy = -(shares * np.log(shares.where(shares > 0))).sum(axis=1, skipna=True)
        columns[f"{family}_volume_log"] = np.log1p(volume).astype("float32")
        columns[f"{family}_reported"] = reported[cols[0]].astype("int8")
        # Bin shares are the "how it was used" signal; volume already lives in the counters.
        for i in range(bins):
            columns[f"{family}_s{i}"] = shares[cols[i]]
        columns[f"{family}_entropy"] = entropy.astype("float32")
        columns[f"{family}_drift"] = (shares - baseline).abs().sum(axis=1, skipna=True).astype("float32")
        columns[f"{family}_entropy_delta"] = (
            entropy - entropy.groupby(frame[VEHICLE_ID], sort=False).transform("first")
        ).astype("float32")

    return pd.concat(columns, axis=1)


def attach_specifications(features: pd.DataFrame, specifications: pd.DataFrame) -> pd.DataFrame:
    specs = specifications.copy()
    for column in SPEC_COLUMNS:
        specs[column] = specs[column].astype("category")
    return features.merge(specs, on=VEHICLE_ID, how="left")


def feature_matrix_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != VEHICLE_ID]
