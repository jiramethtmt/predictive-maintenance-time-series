from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cost import total_cost
from .dataset import EvaluationSet
from .decision import minimum_cost_decision, scaled_cost_matrix
from .io import load_split
from .schema import COST_MATRIX, COUNTERS, TIME_STEP, VEHICLE_ID

SCORING_PAYLOAD = Path(__file__).resolve().parents[2] / "results" / "scoring.json"
TRAJECTORY_VEHICLES_PER_CLASS = 12
TRAJECTORY_COUNTERS = ("171_0", "427_0", "835_0", "100_0")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    table = np.zeros((5, 5), dtype=int)
    np.add.at(table, (y_true.astype(int), y_pred.astype(int)), 1)
    return table.tolist()


def summarise(name: str, y_true: np.ndarray, probabilities: np.ndarray, miss_scale: float) -> dict:
    y_pred = minimum_cost_decision(probabilities, scaled_cost_matrix(miss_scale))
    cost = total_cost(y_true, y_pred)
    baselines = {f"always_{k}": total_cost(y_true, np.full_like(y_true, k)) for k in range(5)}
    baselines["argmax"] = total_cost(y_true, probabilities.argmax(axis=1))
    baselines["untuned"] = total_cost(y_true, minimum_cost_decision(probabilities))
    alerted, at_risk = y_pred > 0, y_true > 0
    caught = int((alerted & at_risk).sum())
    return {
        "split": name,
        "vehicles": int(len(y_true)),
        "miss_scale": miss_scale,
        "total_cost": cost,
        "cost_per_vehicle": round(cost / len(y_true), 3),
        "baselines": baselines,
        "best_baseline": min(v for k, v in baselines.items() if k.startswith("always")),
        "confusion": confusion(y_true, y_pred),
        "alerts": int(alerted.sum()),
        "caught": caught,
        "at_risk": int(at_risk.sum()),
        "recall": round(caught / max(int(at_risk.sum()), 1), 4),
        "precision": round(caught / max(int(alerted.sum()), 1), 4),
    }


def _trajectories(keep: list[int]) -> dict[str, dict]:
    readouts = load_split("test").readouts
    wanted = readouts[readouts[VEHICLE_ID].isin(keep)]
    return {
        str(vehicle_id): {
            "t": [round(float(v), 1) for v in series[TIME_STEP]],
            "counters": {
                c: [None if pd.isna(v) else round(float(v), 1) for v in series[c]]
                for c in TRAJECTORY_COUNTERS
            },
        }
        for vehicle_id, group in wanted.groupby(VEHICLE_ID)
        for series in [group.sort_values(TIME_STEP)]
    }


def _vehicle_records(test: EvaluationSet, probabilities: np.ndarray) -> list[dict]:
    cut = test.cut
    usage = {c: [round(float(v), 3) for v in cut[f"{c}_rate_life"].fillna(0)] for c in COUNTERS}
    return [
        {
            "id": int(vid),
            "y": int(actual),
            "p": [round(float(v), 5) for v in row],
            "t": round(float(t), 1),
            "n": int(n),
            "rates": {c: usage[c][index] for c in COUNTERS},
        }
        for index, (vid, actual, row, t, n) in enumerate(
            zip(cut[VEHICLE_ID], test.truth, probabilities, cut[TIME_STEP], cut["readout_index"] + 1)
        )
    ]


def write_scoring_payload(
    model_name: str,
    miss_scale: float,
    splits: dict[str, tuple[EvaluationSet, np.ndarray]],
) -> Path:
    report = {
        "model": model_name,
        "classes": list(range(5)),
        "cost_matrix": [list(row) for row in COST_MATRIX],
        "miss_scale": miss_scale,
        "splits": {
            name: summarise(name, evaluation.truth, probabilities, miss_scale)
            for name, (evaluation, probabilities) in splits.items()
        },
    }

    test, test_probabilities = splits["test"]
    records = _vehicle_records(test, test_probabilities)
    # The payload keeps raw histories for the highest-risk trucks of each true class, so every
    # class has an inspectable example rather than only the ones the model happened to flag.
    by_class: dict[int, list[int]] = {k: [] for k in range(5)}
    for record in sorted(records, key=lambda r: -max(r["p"][1:])):
        bucket = by_class[record["y"]]
        if len(bucket) < TRAJECTORY_VEHICLES_PER_CLASS:
            bucket.append(record["id"])

    payload = {
        "report": report,
        "vehicles": records,
        "trajectories": _trajectories([vid for bucket in by_class.values() for vid in bucket]),
        "counters": list(COUNTERS),
    }
    SCORING_PAYLOAD.parent.mkdir(parents=True, exist_ok=True)
    SCORING_PAYLOAD.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return SCORING_PAYLOAD
