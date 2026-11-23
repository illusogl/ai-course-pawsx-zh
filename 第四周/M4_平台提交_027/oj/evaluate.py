"""Run course scoring and verify full coverage on all local inputs."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

# Resolve package-relative imports when invoked as python oj/<script>.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "oj"

from .model import USAGE, read_jsonl, write_json
from .scoring import load_meta, score

ROOT = Path(__file__).resolve().parents[1]


def run(split, source, runs):
    target = runs / f"results.{split}.jsonl"
    cmd = [sys.executable, "oj/run.py", "--method", "trad", "--split", split,
           "--input", str(source.resolve()), "--out", str(target.resolve()), "--seed", "0"]
    started = perf_counter()
    process = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    seconds = perf_counter() - started
    command = f"python oj/run.py --method trad --split {split} --input {source.name} --out {target.name} --seed 0"
    (runs / f"{split}.log").write_text(command + "\n" + process.stdout + process.stderr, encoding="utf-8")
    process.check_returncode()
    return read_jsonl(target), round(seconds, 4)


def assert_protocol(inputs, predictions):
    assert [r["item_id"] for r in inputs] == [r["item_id"] for r in predictions]
    assert len({r["item_id"] for r in predictions}) == len(inputs)
    valid = 0
    for row in predictions:
        assert set(row) == {"item_id", "output", "status", "latency_ms", "usage"}
        assert row["usage"] == USAGE and type(row["latency_ms"]) is int and row["latency_ms"] >= 0
        assert row["status"] in {"ok", "fail", "error", "timeout"}
        if row["status"] == "ok":
            assert set(row["output"]) == {"label"}
            label = row["output"]["label"]
            assert type(label) is int and label in (0, 1)
            valid += 1
        else:
            assert row["output"] is None
    return {"rows": len(inputs), "coverage": len(predictions) / len(inputs),
            "valid": valid, "statuses": dict(Counter(r["status"] for r in predictions))}


def interval(correct):
    generator = random.Random(0)
    values = sorted(sum(generator.choices(correct, k=len(correct))) / len(correct)
                    for _ in range(10000))
    return [values[249], values[9749]]


def evaluate(meta, dev, predictions):
    gold = {r["item_id"]: r["target"] for r in dev}
    outputs = {r["item_id"]: r["output"] if r["status"] == "ok" else None for r in predictions}
    metrics = score(meta, gold, [{"id": i, "prediction": value} for i, value in outputs.items()])
    metrics["coverage"] = sum(i in outputs for i in gold) / len(gold)
    metrics["accuracy_ci95"] = interval([outputs.get(i) == expected for i, expected in gold.items()])
    errors = []
    statuses = {r["item_id"]: r["status"] for r in predictions}
    for row in dev:
        item_id = row["item_id"]
        actual = outputs.get(item_id)
        if actual != row["target"]:
            errors.append({"item_id": item_id, "text_pair": row["text_pair"],
                           "gold": row["target"]["label"], "prediction": None if actual is None else actual["label"],
                           "status": statuses.get(item_id, "missing")})
    metrics["error_counts"] = dict(Counter(
        r["status"] if r["status"] != "ok" else "false_positive" if r["prediction"] == 1 else "false_negative"
        for r in errors))
    return metrics, errors


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_hash(predictions):
    digest = hashlib.sha256()
    for row in predictions:
        stable = {key: value for key, value in row.items() if key != "latency_ms"}
        serialized = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update((serialized + "\n").encode("utf-8"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--holdout", action="store_true", help="Run the one-time evaluation after model lock")
    args = parser.parse_args()
    runs = args.out.resolve()
    metrics_path = runs / ("metrics.holdout.json" if args.holdout else "metrics.json")
    if args.holdout and (metrics_path.exists() or (runs / "results.holdout746.jsonl").exists()):
        parser.error("holdout already evaluated in this output directory")

    lock = json.loads((ROOT / "benchmark.json").read_text(encoding="utf-8"))
    for name, expected in lock["files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            parser.error(f"locked SHA-256 mismatch: {name}")
    meta = load_meta(args.data / "metadata.json")
    meta.pop("trivial_baseline", None)
    runs.mkdir(parents=True, exist_ok=True)

    def checked(filename, rows):
        source = args.data / filename
        if hashlib.sha256(source.read_bytes()).hexdigest() != lock["data_sha256"][filename]:
            parser.error(f"frozen SHA-256 mismatch: {filename}")
        items = read_jsonl(source)
        if len(items) != rows:
            parser.error(f"unexpected row count: {filename}")
        return source, items

    def labeled(split, rows):
        source = runs / f"{split}.inputs.jsonl"
        write_rows(source, [{"item_id": row["item_id"], "text_pair": row["text_pair"]} for row in rows])
        try:
            outputs, seconds = run(split, source, runs)
        finally:
            source.unlink(missing_ok=True)
        protocol = assert_protocol(rows, outputs)
        if protocol["valid"] != len(rows):
            raise SystemExit(f"{split} produced failed outputs")
        metrics, errors = evaluate(meta, rows, outputs)
        return metrics, errors, {"sha256": canonical_hash(outputs), "seconds": seconds, **protocol}

    if args.holdout:
        _, rows = checked("splits/holdout746.jsonl", 746)
        metrics, _, protocol = labeled("holdout746", rows)
        result = {"holdout746": metrics, "protocol": protocol}
        print(f"holdout746 accuracy={metrics['accuracy']:.6f} macro_f1={metrics['macro_f1']:.6f}")
    else:
        _, rows = checked("splits/dev1200.jsonl", 1200)
        first, errors, a = labeled("dev1200.a", rows)
        second, _, b = labeled("dev1200.b", rows)
        if first != second or a["sha256"] != b["sha256"]:
            raise SystemExit("dev1200 runs produced different predictions or metrics")
        if first["accuracy"] != lock["selected"]["accuracy"] or abs(first["macro_f1"] - lock["selected"]["macro_f1"]) > 1e-12:
            raise SystemExit("portable scorer does not reproduce the selected trial")
        # Keep IDs and labels as evidence without redistributing the input sentences.
        write_rows(runs / "errors.dev1200.jsonl", [
            {key: value for key, value in row.items() if key != "text_pair"} for row in errors
        ])
        coverage = {}
        for split, filename, size in (("test", "test_inputs.jsonl", 1965), ("compare", "compare50.jsonl", 50),
                                      ("dev100", "dev100.jsonl", 100)):
            source, items = checked(filename, size)
            if split == "dev100":
                source = runs / "dev100.inputs.jsonl"
                write_rows(source, [{"item_id": row["item_id"], "text_pair": row["text_pair"]} for row in items])
            try:
                outputs, seconds = run(split, source, runs)
            finally:
                if split == "dev100":
                    source.unlink(missing_ok=True)
            info = assert_protocol(items, outputs)
            if info["valid"] != size:
                raise SystemExit(f"{split} produced failed outputs")
            coverage[split] = {"seconds": seconds, **info}
        result = {"dev1200": first, "repeat": [a, b], "protocol": coverage}
        print(f"dev1200 accuracy={first['accuracy']:.6f} macro_f1={first['macro_f1']:.6f} hash={a['sha256']}")
    write_json(metrics_path, result)


if __name__ == "__main__":
    main()
