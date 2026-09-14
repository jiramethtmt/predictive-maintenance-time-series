from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import StratifiedGroupKFold

from .dataset import LABEL
from .decision import minimum_cost_decision, scaled_cost_matrix
from .schema import COST_MATRIX, VEHICLE_ID

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


FOLD = "fold"


def assign_folds(pool: pd.DataFrame, folds: int, seed: int) -> pd.Series:
    # AutoGluon's `groups` column is the fold index, split with LeaveOneGroupOut, so the grouping
    # by vehicle has to happen here: cut points from one vehicle are near-duplicates and must not
    # straddle a fold boundary.
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = pd.Series(-1, index=pool.index, dtype="int32")
    for fold, (_, held) in enumerate(splitter.split(pool, pool[LABEL], pool[VEHICLE_ID])):
        assignment.iloc[held] = fold
    return assignment


def fit_predictor(
    pool: pd.DataFrame,
    time_limit: int,
    presets: str,
    # No `optimize_for_deployment` here: its save_space step deletes models/autogluon/utils/data,
    # which is exactly what predict_proba_oof() reads back.
    only: tuple[str, ...] = (),
) -> TabularPredictor:
    predictor = TabularPredictor(
        label=LABEL,
        problem_type="multiclass",
        eval_metric=workshop_cost,
        path=str(MODEL_DIR),
        groups=FOLD,
        verbosity=2,
    )
    # Bagging multiplies every model family by the fold count, so restricting the search to one
    # family is what makes a trend check cheap enough to iterate on.
    restriction = {"included_model_types": list(only)} if only else {}
    return predictor.fit(
        train_data=pool,
        time_limit=time_limit,
        presets=presets,
        excluded_model_types=EXCLUDED_MODELS,
        num_bag_folds=int(pool[FOLD].nunique()),
        **restriction,
    )


def load_predictor() -> TabularPredictor:
    return TabularPredictor.load(str(MODEL_DIR), require_version_match=False)


def predict_all_classes(predictor: TabularPredictor, features: pd.DataFrame) -> np.ndarray:
    proba = predictor.predict_proba(features)
    full = np.zeros((len(features), 5), dtype=float)
    for column in proba.columns:
        full[:, int(column)] = proba[column].to_numpy()
    return full
