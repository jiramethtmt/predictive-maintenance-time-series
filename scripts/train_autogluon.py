import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania import total_cost  # noqa: E402
from scania.dataset import LABEL, evaluation_set, labelled_frame, training_cut_points  # noqa: E402
from scania.decision import minimum_cost_decision, scaled_cost_matrix, tune_miss_scale  # noqa: E402
from scania.ensemble import fit_predictor, predict_all_classes  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def report(name: str, truth: np.ndarray, probabilities: np.ndarray, miss_scale: float) -> None:
    predicted = minimum_cost_decision(probabilities, scaled_cost_matrix(miss_scale))
    alerted, at_risk = predicted > 0, truth > 0
    caught = int((alerted & at_risk).sum())
    print(
        f"{name}: cost {total_cost(truth, predicted)}  alerts {int(alerted.sum())}  "
        f"caught {caught}/{int(at_risk.sum())}  recall {caught / max(int(at_risk.sum()), 1):.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--presets", default="medium_quality")
    parser.add_argument("--bag-folds", type=int, default=0)
    args = parser.parse_args()

    train = training_cut_points().drop(columns=[VEHICLE_ID])
    validation = evaluation_set("validation")
    test = evaluation_set("test")
    print(f"train rows {len(train)}, classes {np.bincount(train[LABEL], minlength=5).tolist()}")

    predictor = fit_predictor(
        train, labelled_frame(validation), args.time_limit, args.presets, args.bag_folds
    )
    print(predictor.leaderboard(labelled_frame(validation), silent=True).head(10))

    validation_proba = predict_all_classes(predictor, validation.cut.drop(columns=[VEHICLE_ID]))
    miss_scale, tuned = tune_miss_scale(validation_proba, validation.truth)
    print(f"miss scale tuned on validation: {miss_scale}x -> {tuned}")

    report("validation", validation.truth, validation_proba, miss_scale)
    test_proba = predict_all_classes(predictor, test.cut.drop(columns=[VEHICLE_ID]))
    report("test", test.truth, test_proba, miss_scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
