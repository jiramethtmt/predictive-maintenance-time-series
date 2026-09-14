import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import LABEL, evaluation_set, labelled_frame, training_cut_points  # noqa: E402
from scania.decision import tune_miss_scale  # noqa: E402
from scania.ensemble import FOLD, assign_folds, fit_predictor, predict_all_classes  # noqa: E402
from scania.report import summarise, write_web_payload  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def build_pool(seed: int, folds: int) -> pd.DataFrame:
    validation = evaluation_set("validation")
    rows = labelled_frame(validation)
    rows[VEHICLE_ID] = validation.cut[VEHICLE_ID].to_numpy()
    pool = pd.concat([training_cut_points(seed=seed), rows], ignore_index=True)
    pool[FOLD] = assign_folds(pool, folds, seed)
    return pool.drop(columns=[VEHICLE_ID])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()

    pool = build_pool(args.seed, args.folds)
    print(f"pool rows {len(pool)}, classes {np.bincount(pool[LABEL], minlength=5).tolist()}")

    predictor = fit_predictor(pool, args.time_limit, args.presets, args.compact)
    print(predictor.leaderboard(silent=True).head(12).to_string())

    # Out-of-fold predictions cover every pooled row, so the decision rule is fitted on far more
    # than the 136 at-risk vehicles a single held-out split would offer.
    oof = predictor.predict_proba_oof().to_numpy()
    miss_scale, tuned = tune_miss_scale(oof, pool[LABEL].to_numpy())
    print(f"\nmiss scale tuned on out-of-fold predictions: {miss_scale}x -> {tuned}")

    test = evaluation_set("test")
    probabilities = predict_all_classes(predictor, test.cut.drop(columns=[VEHICLE_ID]))
    summary = summarise("test", test.truth, probabilities, miss_scale)
    print(
        f"test: cost {summary['total_cost']}  per_vehicle {summary['cost_per_vehicle']}  "
        f"alerts {summary['alerts']}  caught {summary['caught']}/{summary['at_risk']}  "
        f"recall {summary['recall']}  best_baseline {summary['best_baseline']}"
    )

    validation = evaluation_set("validation")
    splits = {
        "validation": (validation, predict_all_classes(predictor, validation.cut.drop(columns=[VEHICLE_ID]))),
        "test": (test, probabilities),
    }
    target = write_web_payload("AutoGluon", miss_scale, splits)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
