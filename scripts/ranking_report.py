import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    payload = json.loads((ROOT / "results" / "scoring.json").read_text(encoding="utf-8"))
    vehicles = payload["vehicles"]
    risk = np.array([1 - v["p"][0] for v in vehicles])
    at_risk = np.array([v["y"] > 0 for v in vehicles])
    ranked = at_risk[np.argsort(-risk)]
    total = int(at_risk.sum())

    print(f"at risk {total} of {len(vehicles)}")
    for k in (50, 100, 200, 500, 1000):
        caught = int(ranked[:k].sum())
        print(f"  top {k:>4}: {caught:>3}/{total}  lift {caught / k / (total / len(vehicles)):.1f}x")

    order = np.argsort(-risk)
    positives = np.flatnonzero(ranked)
    auc = 1 - (positives.mean() / len(vehicles)) if total else float("nan")
    print(f"  mean percentile of a true positive: {positives.mean() / len(vehicles):.3f} (AUC-like {auc:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
