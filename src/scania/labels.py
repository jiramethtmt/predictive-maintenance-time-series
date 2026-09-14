from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import TIME_STEP, VEHICLE_ID

WINDOW_EDGES = (6.0, 12.0, 24.0, 48.0)
CUTS_PER_VEHICLE = 5


def steps_to_class(steps_remaining: np.ndarray) -> np.ndarray:
    # Mirrors the challenge windows: 4 is imminent (0-6 steps), 0 is "more than 48 steps away".
    return 4 - np.searchsorted(WINDOW_EDGES, steps_remaining, side="left")


def label_cut_points(features: pd.DataFrame, tte: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    outcome = tte.set_index(VEHICLE_ID)
    joined = features.join(outcome[["length_of_study_time_step", "in_study_repair"]], on=VEHICLE_ID)
    joined = joined[joined["in_study_repair"].notna()]

    steps_remaining = (joined["length_of_study_time_step"] - joined[TIME_STEP]).to_numpy()
    label = np.where(
        joined["in_study_repair"].to_numpy() == 1,
        steps_to_class(steps_remaining).clip(0, 4),
        0,
    )
    joined["class_label"] = label.astype("int8")

    # Validation and test cut each series at a randomly chosen readout, so sampling cut points
    # uniformly is what keeps the training class prior comparable to the evaluation prior.
    shuffled = joined.sample(frac=1.0, random_state=seed)
    picked = shuffled.groupby(VEHICLE_ID, sort=False).head(CUTS_PER_VEHICLE)
    return picked.drop(columns=["length_of_study_time_step", "in_study_repair"])


def last_readout_per_vehicle(features: pd.DataFrame) -> pd.DataFrame:
    return features.sort_values([VEHICLE_ID, TIME_STEP]).groupby(VEHICLE_ID, sort=False).tail(1)
