import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
PLACEHOLDER = "__PAYLOAD__"


def main() -> int:
    template = (WEB / "app.html").read_text(encoding="utf-8")
    payload = (WEB / "data.json").read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        print(f"{PLACEHOLDER} missing from app.html", file=sys.stderr)
        return 1
    # A literal </script> inside the JSON would close the host tag early.
    page = template.replace(PLACEHOLDER, payload.replace("</", "<\\/"))
    target = WEB / "index.html"
    target.write_text(page, encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
