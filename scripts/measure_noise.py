import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import evaluation_set, training_cut_points  # noqa: E402
from scania.decision import tune_miss_scale  # noqa: E402
from scania.model import build_classifier, predict_all_classes, training_matrix  # noqa: E402
from scania.report import summarise  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    splits = {name: evaluation_set(name) for name in ("validation", "test")}
    records = []
    for seed in range(args.runs):
        x_train, y_train = training_matrix(training_cut_points(seed=seed))
        model = build_classifier(seed=seed)
        model.fit(x_train, y_train)

        probabilities = {
            name: predict_all_classes(model, evaluation.cut.drop(columns=[VEHICLE_ID]))
            for name, evaluation in splits.items()
        }
        miss_scale, _ = tune_miss_scale(probabilities["validation"], splits["validation"].truth)
        row = {"seed": seed, "miss_scale": miss_scale}
        for name, evaluation in splits.items():
            summary = summarise(name, evaluation.truth, probabilities[name], miss_scale)
            row[f"{name}_cost"] = summary["total_cost"]
            row[f"{name}_recall"] = summary["recall"]
        records.append(row)
        print(row, flush=True)

    frame = pd.DataFrame(records)
    print("\nper-run results")
    print(frame.to_string(index=False))
    print("\nspread across seeds, nothing changed but the random state")
    for column in ("validation_cost", "test_cost"):
        values = frame[column].to_numpy()
        print(
            f"  {column}: mean {values.mean():.0f}  std {values.std(ddof=1):.0f}  "
            f"min {values.min()}  max {values.max()}  range {values.max() - values.min()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
