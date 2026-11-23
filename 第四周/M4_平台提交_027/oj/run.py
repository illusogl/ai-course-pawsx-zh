"""Offline M4 inference entry point."""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

# Resolve package-relative imports when invoked as python oj/<script>.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "oj"

from .model import BASE_FEATURE_NAMES, FEATURE_NAMES, USAGE, classify, features

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["trad"], default="trad")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.input.resolve() == args.out.resolve():
        parser.error("input and output must be different files")

    model_error = None
    try:
        model = json.loads((ROOT / "models/trad.json").read_text(encoding="utf-8"))
        if model["features"] not in (BASE_FEATURE_NAMES, FEATURE_NAMES):
            raise ValueError("model feature schema mismatch")
    except (OSError, ValueError, KeyError, TypeError) as error:
        model_error = str(error)
        print(f"model load failed: {error}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = failures = unidentified = 0
    with args.input.open(encoding="utf-8-sig") as source, args.out.open("w", encoding="utf-8") as output:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            started = perf_counter()
            try:
                row = json.loads(line)
                item_id = row["item_id"]
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError("missing or invalid item_id")
            except (ValueError, KeyError, TypeError) as error:
                unidentified += 1
                print(f"line {number}: {error}", file=sys.stderr)
                continue

            prediction = None
            status = "ok"
            try:
                if model_error is not None:
                    raise ValueError("model unavailable")
                prediction = {"label": classify(model, features(row["text_pair"]))}
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                status = "error"
                failures += 1
                print(f"line {number}, item_id={item_id!r}: {error}", file=sys.stderr)
            result = {
                "item_id": item_id, "output": prediction, "status": status,
                "latency_ms": int((perf_counter() - started) * 1000), "usage": USAGE,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            count += 1
    print(f"trad {args.split}: rows={count}, errors={failures}, unidentified={unidentified}")
    return 1 if unidentified or model_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
