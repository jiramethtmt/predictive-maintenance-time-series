from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .schema import SPEC_COLUMNS, VEHICLE_ID

CLASS_ORDER = np.arange(5)


def training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    features = frame.drop(columns=[VEHICLE_ID, "class_label"])
    return features, frame["class_label"].to_numpy()


def build_classifier(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        categorical_features=list(SPEC_COLUMNS),
        random_state=seed,
    )


def predict_all_classes(model: HistGradientBoostingClassifier, features: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(features)
    # A cut-point sample can miss a rare class entirely; the cost rule needs all five columns.
    full = np.zeros((len(features), len(CLASS_ORDER)), dtype=float)
    full[:, model.classes_.astype(int)] = raw
    return full
