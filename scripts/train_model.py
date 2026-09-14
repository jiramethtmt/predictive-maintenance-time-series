import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania import COST_MATRIX, COUNTERS, TIME_STEP, VEHICLE_ID, load_split, total_cost  # noqa: E402
from scania.decision import minimum_cost_decision  # noqa: E402
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


def evaluate(name: str, y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = minimum_cost_decision(probabilities)
    cost = total_cost(y_true, y_pred)
    baselines = {
        f"always_{k}": total_cost(y_true, np.full_like(y_true, k)) for k in range(5)
    }
    baselines["argmax"] = total_cost(y_true, probabilities.argmax(axis=1))
    return {
        "split": name,
        "vehicles": int(len(y_true)),
        "total_cost": cost,
        "cost_per_vehicle": round(cost / len(y_true), 3),
        "baselines": baselines,
        "best_baseline": min(baselines.values()),
        "confusion": confusion(y_true, y_pred),
        "alerts": int((y_pred > 0).sum()),
        "caught": int(((y_true > 0) & (y_pred > 0)).sum()),
        "at_risk": int((y_true > 0).sum()),
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

    report = {"classes": list(range(5)), "cost_matrix": [list(row) for row in COST_MATRIX], "splits": {}}
    vehicle_records: list[dict] = []

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
        report["splits"][split_name] = evaluate(split_name, truth, probabilities)
        print(split_name, report["splits"][split_name]["total_cost"], report["splits"][split_name]["baselines"])

        if split_name == "test":
            vehicle_records = [
                {
                    "id": int(vid),
                    "y": int(actual),
                    "p": [round(float(v), 5) for v in row],
                    "t": round(float(t), 1),
                    "n": int(n),
                }
                for vid, actual, row, t, n in zip(
                    cut[VEHICLE_ID], truth, probabilities, cut[TIME_STEP], cut["readout_index"] + 1
                )
            ]
            usage = {
                c: [round(float(v), 3) for v in cut[f"{c}_rate_life"].fillna(0)] for c in COUNTERS
            }
            for index, record in enumerate(vehicle_records):
                record["rates"] = {c: usage[c][index] for c in COUNTERS}

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
