import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import LABEL, evaluation_set, labelled_frame, training_cut_points  # noqa: E402
from scania.decision import tune_miss_scale  # noqa: E402
from scania.model import build_classifier, predict_all_classes  # noqa: E402
from scania.report import summarise  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def pooled_training_rows(seed: int, include_validation: bool) -> pd.DataFrame:
    rows = [training_cut_points(seed=seed)]
    if include_validation:
        validation = evaluation_set("validation")
        pooled = labelled_frame(validation)
        pooled[VEHICLE_ID] = validation.cut[VEHICLE_ID].to_numpy()
        rows.append(pooled)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    include_validation = not args.no_validation
    records = []
    for seed in range(args.seeds):
        pool = pooled_training_rows(seed, include_validation)
        groups = pool[VEHICLE_ID].to_numpy()
        y = pool[LABEL].to_numpy()
        features = pool.drop(columns=[VEHICLE_ID, LABEL])

        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold, (fit_idx, held_idx) in enumerate(splitter.split(features, y, groups)):
            overlap = set(groups[fit_idx]) & set(groups[held_idx])
            if overlap:
                raise AssertionError(f"{len(overlap)} vehicles appear on both sides of fold {fold}")

            model = build_classifier(seed=seed)
            model.fit(features.iloc[fit_idx], y[fit_idx])
            probabilities = predict_all_classes(model, features.iloc[held_idx])
            miss_scale, _ = tune_miss_scale(probabilities, y[held_idx])
            summary = summarise("fold", y[held_idx], probabilities, miss_scale)
            records.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "held_rows": len(held_idx),
                    "held_at_risk": summary["at_risk"],
                    "cost_per_vehicle": summary["cost_per_vehicle"],
                    "recall": summary["recall"],
                }
            )
            print(records[-1], flush=True)

    frame = pd.DataFrame(records)
    print("\nper-fold results")
    print(frame.to_string(index=False))
    values = frame["cost_per_vehicle"].to_numpy()
    print(
        f"\npooled validation into training: {include_validation}"
        f"\ncost per vehicle: mean {values.mean():.3f}  std {values.std(ddof=1):.3f}"
        f"  min {values.min():.3f}  max {values.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
