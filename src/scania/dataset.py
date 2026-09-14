from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import attach_specifications, build_row_features
from .io import load_split
from .labels import label_cut_points, last_readout_per_vehicle
from .schema import VEHICLE_ID

LABEL = "class_label"


@dataclass(frozen=True)
class EvaluationSet:
    name: str
    cut: pd.DataFrame
    truth: np.ndarray


def split_features(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = load_split(name)
    return attach_specifications(build_row_features(split.readouts), split.specifications), split.labels


def training_cut_points() -> pd.DataFrame:
    features, tte = split_features("train")
    return label_cut_points(features, tte)


def evaluation_set(name: str) -> EvaluationSet:
    features, labels = split_features(name)
    cut = last_readout_per_vehicle(features)
    truth = labels.set_index(VEHICLE_ID).loc[cut[VEHICLE_ID], LABEL].to_numpy().astype(int)
    return EvaluationSet(name=name, cut=cut, truth=truth)


def labelled_frame(evaluation: EvaluationSet) -> pd.DataFrame:
    frame = evaluation.cut.drop(columns=[VEHICLE_ID]).copy()
    frame[LABEL] = evaluation.truth
    return frame
