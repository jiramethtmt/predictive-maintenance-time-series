from __future__ import annotations

VEHICLE_ID = "vehicle_id"
TIME_STEP = "time_step"

COUNTERS: tuple[str, ...] = (
    "171_0",
    "666_0",
    "427_0",
    "837_0",
    "309_0",
    "835_0",
    "370_0",
    "100_0",
)

HISTOGRAM_BINS: dict[str, int] = {
    "167": 10,
    "272": 10,
    "291": 11,
    "158": 10,
    "459": 20,
    "397": 36,
}

SPEC_COLUMNS: tuple[str, ...] = tuple(f"Spec_{i}" for i in range(8))

# IDA 2024 challenge cost matrix, rows = actual class, cols = predicted class.
# Missing a real failure (rows 1-4 predicted as 0) costs 20-70x an unnecessary workshop check.
COST_MATRIX: tuple[tuple[int, ...], ...] = (
    (0, 7, 8, 9, 10),
    (200, 0, 7, 8, 9),
    (300, 200, 0, 7, 8),
    (400, 300, 200, 0, 7),
    (500, 400, 300, 200, 0),
)

CLASS_WINDOWS: dict[int, str] = {
    0: "> 48 steps to failure (or healthy)",
    1: "48-24 steps to failure",
    2: "24-12 steps to failure",
    3: "12-6 steps to failure",
    4: "6-0 steps to failure",
}


def histogram_columns(family: str) -> list[str]:
    return [f"{family}_{i}" for i in range(HISTOGRAM_BINS[family])]


def all_histogram_columns() -> list[str]:
    return [c for family in HISTOGRAM_BINS for c in histogram_columns(family)]


def feature_columns() -> list[str]:
    return list(COUNTERS) + all_histogram_columns()
