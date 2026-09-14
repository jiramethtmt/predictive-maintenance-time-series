from __future__ import annotations

import numpy as np

from .schema import COST_MATRIX

_COST = np.asarray(COST_MATRIX, dtype=np.int64)


def total_cost(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    return int(_COST[np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)].sum())


def cost_per_vehicle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    return total_cost(y_true, y_pred) / len(y_true)


def constant_prediction_costs(y_true: np.ndarray) -> dict[int, int]:
    y_true = np.asarray(y_true, dtype=int)
    return {k: total_cost(y_true, np.full_like(y_true, k)) for k in range(_COST.shape[1])}
