import sys
import urllib.request
from pathlib import Path

BASE = "https://doris.snd.se/api/file/2024-34/3"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

DATA_FILES = [
    "train_operational_readouts.csv",
    "train_tte.csv",
    "train_specifications.csv",
    "validation_operational_readouts.csv",
    "validation_labels.csv",
    "validation_specifications.csv",
    "test_operational_readouts.csv",
    "test_labels.csv",
    "test_specifications.csv",
]
DOC_FILES = ["Scania_Component_X.pdf", "2024_IDA_challenge_v2.pdf"]


def fetch(kind: str, name: str) -> None:
    target = RAW / name
    if target.exists() and target.stat().st_size > 0:
        print(f"skip   {name} ({target.stat().st_size / 1e6:.1f} MB)")
        return
    url = f"{BASE}/{kind}?filePath={name}"
    print(f"get    {name}", flush=True)
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    print(f"done   {name} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        fetch("data", name)
    for name in DOC_FILES:
        fetch("documentation", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
