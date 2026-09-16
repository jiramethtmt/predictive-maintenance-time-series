import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import evaluation_set, training_cut_points  # noqa: E402
from scania.decision import tune_miss_scale  # noqa: E402
from scania.model import build_classifier, predict_all_classes, training_matrix  # noqa: E402
from scania.report import summarise, write_scoring_payload  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def main() -> int:
    cut_points = training_cut_points()
    x_train, y_train = training_matrix(cut_points)
    print(f"train cut points {len(x_train)}, class counts {np.bincount(y_train, minlength=5).tolist()}")

    model = build_classifier()
    model.fit(x_train, y_train)
    print(f"fitted, iterations {model.n_iter_}")

    splits = {}
    for name in ("validation", "test"):
        evaluation = evaluation_set(name)
        probabilities = predict_all_classes(model, evaluation.cut.drop(columns=[VEHICLE_ID]))
        splits[name] = (evaluation, probabilities)

    validation, validation_probabilities = splits["validation"]
    miss_scale, tuned = tune_miss_scale(validation_probabilities, validation.truth)
    print(f"miss scale tuned on validation: {miss_scale}x -> {tuned}")

    for name, (evaluation, probabilities) in splits.items():
        summary = summarise(name, evaluation.truth, probabilities, miss_scale)
        print(f"{name}: cost {summary['total_cost']}  recall {summary['recall']}  {summary['baselines']}")

    target = write_scoring_payload("HistGradientBoosting", miss_scale, splits)
    print(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
