import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania import COST_MATRIX, COUNTERS, TIME_STEP, VEHICLE_ID, load_split, total_cost  # noqa: E402
from scania.decision import minimum_cost_decision, scaled_cost_matrix, tune_miss_scale  # noqa: E402
from scania.features import attach_specifications, build_row_features  # noqa: E402
from scania.labels import label_cut_points, last_readout_per_vehicle  # noqa: E402
from scania.model import build_classifier, predict_all_classes, training_matrix  # noqa: E402

WEB_DATA = ROOT / "web" / "data.json"
TRAJECTORY_VEHICLES_PER_CLASS = 12
TRAJECTORY_COUNTERS = ("171_0", "427_0", "835_0", "100_0")


def prepare(split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = load_split(split_name)
    features = attach_specifications(build_row_features(split.readouts), split.specifications)
    return features, split.labels


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    table = np.zeros((5, 5), dtype=int)
    np.add.at(table, (y_true.astype(int), y_pred.astype(int)), 1)
    return table.tolist()


def evaluate(name: str, y_true: np.ndarray, probabilities: np.ndarray, miss_scale: float) -> dict:
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


def trajectories(split_name: str, keep: list[int]) -> dict[str, dict]:
    split = load_split(split_name)
    wanted = split.readouts[split.readouts[VEHICLE_ID].isin(keep)]
    out = {}
    for vehicle_id, group in wanted.groupby(VEHICLE_ID):
        series = group.sort_values(TIME_STEP)
        out[str(vehicle_id)] = {
            "t": [round(float(v), 1) for v in series[TIME_STEP]],
            "counters": {
                c: [None if pd.isna(v) else round(float(v), 1) for v in series[c]]
                for c in TRAJECTORY_COUNTERS
            },
        }
    return out


def main() -> int:
    train_features, train_tte = prepare("train")
    cut_points = label_cut_points(train_features, train_tte)
    x_train, y_train = training_matrix(cut_points)
    print(f"train cut points {len(x_train)}, class counts {np.bincount(y_train, minlength=5).tolist()}")

    model = build_classifier()
    model.fit(x_train, y_train)
    print(f"fitted, iterations {model.n_iter_}")

    scored = {}
    for split_name in ("validation", "test"):
        features, labels = prepare(split_name)
        cut = last_readout_per_vehicle(features)
        probabilities = predict_all_classes(model, cut.drop(columns=[VEHICLE_ID]))
        truth = (
            labels.set_index(VEHICLE_ID)
            .loc[cut[VEHICLE_ID], "class_label"]
            .to_numpy()
            .astype(int)
        )
        scored[split_name] = (cut, probabilities, truth)

    miss_scale, tuned_validation_cost = tune_miss_scale(*scored["validation"][1:])
    print(f"miss scale tuned on validation: {miss_scale}x -> {tuned_validation_cost}")

    report = {
        "classes": list(range(5)),
        "cost_matrix": [list(row) for row in COST_MATRIX],
        "miss_scale": miss_scale,
        "splits": {},
    }
    for split_name, (_, probabilities, truth) in scored.items():
        summary = evaluate(split_name, truth, probabilities, miss_scale)
        report["splits"][split_name] = summary
        print(split_name, summary["total_cost"], summary["baselines"], "recall", summary["recall"])

    cut, probabilities, truth = scored["test"]
    usage = {c: [round(float(v), 3) for v in cut[f"{c}_rate_life"].fillna(0)] for c in COUNTERS}
    vehicle_records = [
        {
            "id": int(vid),
            "y": int(actual),
            "p": [round(float(v), 5) for v in row],
            "t": round(float(t), 1),
            "n": int(n),
            "rates": {c: usage[c][index] for c in COUNTERS},
        }
        for index, (vid, actual, row, t, n) in enumerate(
            zip(cut[VEHICLE_ID], truth, probabilities, cut[TIME_STEP], cut["readout_index"] + 1)
        )
    ]

    by_class: dict[int, list[int]] = {k: [] for k in range(5)}
    for record in sorted(vehicle_records, key=lambda r: -max(r["p"][1:])):
        bucket = by_class[record["y"]]
        if len(bucket) < TRAJECTORY_VEHICLES_PER_CLASS:
            bucket.append(record["id"])
    keep = [vid for bucket in by_class.values() for vid in bucket]

    payload = {
        "report": report,
        "vehicles": vehicle_records,
        "trajectories": trajectories("test", keep),
        "counters": list(COUNTERS),
    }
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {WEB_DATA} ({WEB_DATA.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
