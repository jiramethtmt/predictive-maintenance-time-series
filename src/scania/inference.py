from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from .dataset import evaluation_set
from .decision import expected_cost, minimum_cost_decision, scaled_cost_matrix
from .ensemble import MODEL_DIR, predict_all_classes
from .schema import CLASS_WINDOWS, COST_MATRIX, VEHICLE_ID

TUNED_MISS_SCALE = 2.0


@dataclass(frozen=True)
class Verdict:
    probabilities: list[float]
    risk: float
    expected_costs: list[float]
    action: int


class Engine:
    def __init__(self, split: str = "test", miss_scale: float = TUNED_MISS_SCALE) -> None:
        self.predictor = TabularPredictor.load(str(MODEL_DIR), require_version_match=False)
        self.miss_scale = miss_scale
        evaluation = evaluation_set(split)
        self.vehicle_ids = evaluation.cut[VEHICLE_ID].to_numpy()
        self.truth = evaluation.truth
        self.features = evaluation.cut.drop(columns=[VEHICLE_ID]).reset_index(drop=True)
        self._row_of = {int(vehicle): row for row, vehicle in enumerate(self.vehicle_ids)}
        self.baseline = predict_all_classes(self.predictor, self.features)

    @property
    def numeric_columns(self) -> list[str]:
        return [c for c in self.features.columns if pd.api.types.is_numeric_dtype(self.features[c])]

    def row_of(self, vehicle_id: int) -> int:
        row = self._row_of.get(int(vehicle_id))
        if row is None:
            raise KeyError(f"vehicle {vehicle_id} is not in this split")
        return row

    def verdict(self, probabilities: np.ndarray, miss_scale: float | None = None) -> Verdict:
        scale = self.miss_scale if miss_scale is None else miss_scale
        costs = expected_cost(probabilities.reshape(1, -1), scaled_cost_matrix(scale))[0]
        action = int(minimum_cost_decision(probabilities.reshape(1, -1), scaled_cost_matrix(scale))[0])
        return Verdict(
            probabilities=[float(p) for p in probabilities],
            risk=float(1.0 - probabilities[0]),
            expected_costs=[float(c) for c in costs],
            action=action,
        )

    def score_stored(self, vehicle_id: int, miss_scale: float | None = None) -> Verdict:
        return self.verdict(self.baseline[self.row_of(vehicle_id)], miss_scale)

    def score_modified(
        self, vehicle_id: int, overrides: dict[str, float], miss_scale: float | None = None
    ) -> Verdict:
        # A real forward pass through the fitted predictor, not a lookup: this is what makes the
        # what-if honest when a counter's wear rate is nudged.
        frame = self.features.iloc[[self.row_of(vehicle_id)]].copy()
        for column, value in overrides.items():
            if column not in frame.columns:
                raise KeyError(f"unknown feature {column}")
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise TypeError(f"{column} is not numeric")
            frame.loc[frame.index[0], column] = float(value)
        return self.verdict(predict_all_classes(self.predictor, frame)[0], miss_scale)

    def fleet_summary(self, miss_scale: float | None = None) -> dict:
        scale = self.miss_scale if miss_scale is None else miss_scale
        actions = minimum_cost_decision(self.baseline, scaled_cost_matrix(scale))
        real = np.asarray(COST_MATRIX, dtype=float)
        at_risk = self.truth > 0
        return {
            "split_size": int(len(self.truth)),
            "at_risk": int(at_risk.sum()),
            "alerts": int((actions > 0).sum()),
            "caught": int((at_risk & (actions > 0)).sum()),
            "total_cost": int(real[self.truth.astype(int), actions].sum()),
            "miss_scale": float(scale),
        }

    def ranked(self, limit: int, miss_scale: float | None = None) -> list[dict]:
        scale = self.miss_scale if miss_scale is None else miss_scale
        actions = minimum_cost_decision(self.baseline, scaled_cost_matrix(scale))
        risk = 1.0 - self.baseline[:, 0]
        order = np.argsort(-risk)[:limit]
        return [
            {
                "vehicle_id": int(self.vehicle_ids[row]),
                "risk": round(float(risk[row]), 4),
                "true_class": int(self.truth[row]),
                "true_window": CLASS_WINDOWS[int(self.truth[row])],
                "called": bool(actions[row] > 0),
            }
            for row in order
        ]