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
