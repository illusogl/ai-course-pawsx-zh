"""Run a protocol smoke test on dev100."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Resolve package-relative imports when invoked as python oj/<script>.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "oj"

from . import scoring
from .model import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="trad")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/dev100.jsonl")
    parser.add_argument("--entry", type=Path, default=ROOT / "oj/run.py")
    parser.add_argument("--meta", type=Path, default=ROOT / "data/metadata.json")
    args = parser.parse_args()

    rows = read_jsonl(args.dev)
    meta = scoring.load_meta(args.meta)
    meta.pop("trivial_baseline", None)
    gold = {row["item_id"]: row["target"] for row in rows}

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "items.jsonl"
        destination = Path(directory) / "results.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for row in rows:
                item = {key: value for key, value in row.items() if key != "target"}
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        process = subprocess.run(
            [sys.executable, str(args.entry.resolve()), "--method", args.method, "--split", "dev",
             "--input", str(source), "--out", str(destination), "--seed", "0"],
            cwd=ROOT, capture_output=True, encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        results = read_jsonl(destination) if destination.exists() else []

    predictions = [{"id": row.get("item_id"),
                    "prediction": row.get("output") if row.get("status") == "ok" else None}
                   for row in results]
    metrics = scoring.score(meta, gold, predictions)
    print("=== M4 dev100 协议自检 ===")
    print(f"题目数 {metrics['total']} | 结果数 {metrics['prediction_records']} | "
          f"缺失 {metrics['missing']} | 重复 {metrics['duplicate_ids']} | 未知 ID {metrics['unknown_records']}")
    print(f"合法输出 {metrics['valid_output_count']}/{metrics['total']} ({metrics['valid_output_rate']:.1%})")
    if process.stderr:
        print(process.stderr, file=sys.stderr, end="")
    return int(process.returncode != 0 or metrics["valid_output_rate"] != 1
               or metrics["duplicate_ids"] != 0 or metrics["unknown_records"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
