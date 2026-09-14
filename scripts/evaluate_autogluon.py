import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import evaluation_set, labelled_frame  # noqa: E402
from scania.decision import tune_miss_scale  # noqa: E402
from scania.ensemble import load_predictor, predict_all_classes  # noqa: E402
from scania.report import summarise  # noqa: E402
from scania.schema import VEHICLE_ID  # noqa: E402


def main() -> int:
    predictor = load_predictor()
    validation = evaluation_set("validation")
    test = evaluation_set("test")

    print(predictor.leaderboard(labelled_frame(validation), silent=True).head(20).to_string())

    validation_proba = predict_all_classes(predictor, validation.cut.drop(columns=[VEHICLE_ID]))
    miss_scale, tuned = tune_miss_scale(validation_proba, validation.truth)
    print(f"\nmiss scale tuned on validation: {miss_scale}x -> {tuned}")

    for evaluation, probabilities in (
        (validation, validation_proba),
        (test, predict_all_classes(predictor, test.cut.drop(columns=[VEHICLE_ID]))),
    ):
        s = summarise(evaluation.name, evaluation.truth, probabilities, miss_scale)
        print(
            f"{s['split']}: cost {s['total_cost']}  per_vehicle {s['cost_per_vehicle']}  "
            f"alerts {s['alerts']}  caught {s['caught']}/{s['at_risk']}  "
            f"recall {s['recall']}  precision {s['precision']}  best_baseline {s['best_baseline']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
