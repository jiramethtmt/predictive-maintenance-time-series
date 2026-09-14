from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor

from .dataset import LABEL
from .decision import minimum_cost_decision, scaled_cost_matrix
from .schema import COST_MATRIX

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "autogluon"
# Model selection has to see the same asymmetry the workshop does. Selecting on log loss or
# accuracy picks the model that never alarms, which is exactly the failure mode being fixed.
SELECTION_MISS_SCALE = 2.0
# Without ray installed AutoGluon fits bagged folds sequentially, and the neural families are
# handed the majority of the time budget before a single tree model runs. On 93k tabular rows
# the trees are what win, so the budget goes to them.
EXCLUDED_MODELS: tuple[str, ...] = ("NN_TORCH", "FASTAI", "KNN")
# ray is unavailable for this interpreter, so bagged folds fit one after another and a bagged
# preset spends its whole budget on the first model. Bagging is therefore opt-in, not default.


def _negative_total_cost(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    probabilities = np.asarray(y_pred_proba, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1 - probabilities, probabilities])
    if probabilities.shape[1] < 5:
        padded = np.zeros((len(probabilities), 5))
        padded[:, : probabilities.shape[1]] = probabilities
        probabilities = padded
    predicted = minimum_cost_decision(probabilities, scaled_cost_matrix(SELECTION_MISS_SCALE))
    cost = np.asarray(COST_MATRIX, dtype=float)[np.asarray(y_true, dtype=int), predicted]
    return -float(cost.mean())


workshop_cost = make_scorer(
    name="workshop_cost",
    score_func=_negative_total_cost,
    optimum=0,
    greater_is_better=True,
    needs_proba=True,
)


def fit_predictor(
    train: pd.DataFrame,
    tuning: pd.DataFrame,
    time_limit: int,
    presets: str,
    compact: bool,
    bag_folds: int,
) -> TabularPredictor:
    predictor = TabularPredictor(
        label=LABEL,
        problem_type="multiclass",
        eval_metric=workshop_cost,
        path=str(MODEL_DIR),
        verbosity=2,
    )
    bagging = {"num_bag_folds": bag_folds, "num_bag_sets": 1, "use_bag_holdout": True} if bag_folds else {}
    # Cut points from one vehicle are near-duplicates, so an internal random split would leak.
    # The real validation split is vehicle-disjoint by construction; hand it over as tuning_data.
    return predictor.fit(
        train_data=train,
        tuning_data=tuning,
        time_limit=time_limit,
        presets=[presets, "optimize_for_deployment"] if compact else presets,
        excluded_model_types=EXCLUDED_MODELS,
        **bagging,
    )


def load_predictor() -> TabularPredictor:
    return TabularPredictor.load(str(MODEL_DIR), require_version_match=False)


def predict_all_classes(predictor: TabularPredictor, features: pd.DataFrame) -> np.ndarray:
    proba = predictor.predict_proba(features)
    full = np.zeros((len(features), 5), dtype=float)
    for column in proba.columns:
        full[:, int(column)] = proba[column].to_numpy()
    return full
