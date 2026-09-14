from __future__ import annotations

import numpy as np

from .schema import COST_MATRIX


def expected_cost(probabilities: np.ndarray, cost_matrix: np.ndarray | None = None) -> np.ndarray:
    cost = np.asarray(COST_MATRIX if cost_matrix is None else cost_matrix, dtype=float)
    return probabilities @ cost


def minimum_cost_decision(probabilities: np.ndarray, cost_matrix: np.ndarray | None = None) -> np.ndarray:
    # Bayes decision under the challenge's asymmetric loss: pick the action with the lowest
    # expected cost, not the most likely class. Argmax would almost never raise an alarm.
    return expected_cost(probabilities, cost_matrix).argmin(axis=1).astype("int8")


MISS_SCALE_GRID: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0)


def scaled_cost_matrix(miss_scale: float) -> np.ndarray:
    cost = np.asarray(COST_MATRIX, dtype=float)
    return np.where(np.tril(np.ones_like(cost), -1) > 0, cost * miss_scale, cost)


def tune_miss_scale(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[float, int]:
    # The Bayes rule is only optimal for calibrated probabilities. A gradient boosting model
    # trained at a 2.7% positive rate is under-confident on the rare classes, so the decision
    # boundary sits too far toward "do nothing". Inflating the miss cost on a held-out split
    # moves it back; the multiplier is fitted, never taken from the evaluation split.
    real = np.asarray(COST_MATRIX, dtype=float)
    best_scale, best_cost = 1.0, None
    for scale in MISS_SCALE_GRID:
        predicted = minimum_cost_decision(probabilities, scaled_cost_matrix(scale))
        cost = int(real[y_true.astype(int), predicted].sum())
        if best_cost is None or cost < best_cost:
            best_scale, best_cost = scale, cost
    return best_scale, int(best_cost)
