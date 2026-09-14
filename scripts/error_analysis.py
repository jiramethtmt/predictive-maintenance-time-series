import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.dataset import training_cut_points  # noqa: E402
from scania.io import load_split  # noqa: E402
from scania.labels import WINDOW_EDGES  # noqa: E402


def window_widths() -> dict[int, float]:
    edges = (0.0,) + WINDOW_EDGES
    return {4 - i: edges[i + 1] - edges[i] for i in range(len(WINDOW_EDGES))}


def prior_table() -> pd.DataFrame:
    train = training_cut_points()["class_label"].value_counts().sort_index()
    rows = {"train_cuts": train}
    for split in ("validation", "test"):
        rows[split] = load_split(split).labels["class_label"].value_counts().sort_index()

    table = pd.DataFrame(rows).fillna(0).astype(int)
    positives = table.loc[1:]
    share = positives / positives.sum()
    ratio = positives.div(positives.loc[4])
    return table, share.round(3), ratio.round(2)


def per_class_recall() -> pd.DataFrame | None:
    payload_path = ROOT / "web" / "data.json"
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    confusion = np.array(payload["report"]["splits"]["test"]["confusion"])
    alerted = confusion[:, 1:].sum(axis=1)
    total = confusion.sum(axis=1)
    risk = np.array([1 - v["p"][0] for v in payload["vehicles"]])
    truth = np.array([v["y"] for v in payload["vehicles"]])
    order = np.argsort(-risk)
    rank_of_positive = {
        k: np.flatnonzero(truth[order] == k).mean() / len(truth) for k in range(1, 5)
    }
    return pd.DataFrame(
        {
            "vehicles": pd.Series(total, index=range(5)),
            "alerted": pd.Series(alerted, index=range(5)),
            "recall": pd.Series((alerted / np.maximum(total, 1)).round(3), index=range(5)),
            "mean_risk_percentile": pd.Series(rank_of_positive).round(3),
        }
    )


def main() -> int:
    table, share, ratio = prior_table()
    print("class counts")
    print(table.to_string())
    print("\nshare of the positive classes only")
    print(share.to_string())
    print("\nratio against class 4 (imminent)")
    print(ratio.to_string())
    print("\nwindow widths in time_step")
    print(pd.Series(window_widths()).to_string())

    recall = per_class_recall()
    if recall is not None:
        print("\nper-class behaviour on test, current payload model")
        print(recall.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
