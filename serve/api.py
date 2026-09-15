from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scania.inference import Engine  # noqa: E402
from scania.schema import CLASS_WINDOWS, COST_MATRIX  # noqa: E402

PAGE = (Path(__file__).resolve().parent / "index.html").read_bytes()
ENGINE: Engine | None = None


def as_float(values: dict, key: str) -> float | None:
    raw = values.get(key)
    return float(raw[0]) if raw else None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:
        pass

    def send_json(self, payload: dict | list, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        query = parse_qs(url.query)
        miss = as_float(query, "miss_scale")
        try:
            if url.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
            elif url.path == "/api/summary":
                self.send_json(
                    {
                        "fleet": ENGINE.fleet_summary(miss),
                        "cost_matrix": [list(row) for row in COST_MATRIX],
                        "class_windows": {str(k): v for k, v in CLASS_WINDOWS.items()},
                    }
                )
            elif url.path == "/api/ranked":
                limit = int(as_float(query, "limit") or 40)
                self.send_json(ENGINE.ranked(min(limit, 500), miss))
            elif url.path == "/api/features":
                self.send_json(ENGINE.numeric_columns)
            elif url.path.startswith("/api/vehicle/"):
                vehicle = int(url.path.rsplit("/", 1)[1])
                row = ENGINE.row_of(vehicle)
                verdict = ENGINE.score_stored(vehicle, miss)
                self.send_json(
                    {
                        "vehicle_id": vehicle,
                        "true_class": int(ENGINE.truth[row]),
                        "verdict": verdict.__dict__,
                        "features": {
                            column: (None if ENGINE.features[column].isna().iloc[row] else float(ENGINE.features[column].iloc[row]))
                            for column in ENGINE.numeric_columns
                        },
                    }
                )
            else:
                self.send_json({"error": "not found"}, 404)
        except (KeyError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/whatif":
            self.send_json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            verdict = ENGINE.score_modified(
                int(body["vehicle_id"]), body.get("overrides", {}), body.get("miss_scale")
            )
            self.send_json(verdict.__dict__)
        except (KeyError, TypeError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)


def main() -> int:
    global ENGINE
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8800
    print("loading predictor and test split", flush=True)
    ENGINE = Engine()
    print(f"ready on 0.0.0.0:{port} with {len(ENGINE.truth)} trucks", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())