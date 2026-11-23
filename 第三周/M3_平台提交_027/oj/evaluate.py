"""Run course scoring and verify full coverage on all local inputs."""
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

from .model import USAGE, features, read_jsonl, write_json
from .scoring import load_meta, score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs/trad"


def run(split, source):
    target = RUNS / f"results.{split}.jsonl"
    cmd = [sys.executable, "oj/run.py", "--method", "trad", "--split", split,
           "--input", str(source.relative_to(ROOT)), "--out", str(target.relative_to(ROOT)), "--seed", "0"]
    process = subprocess.run(cmd, cwd=ROOT, capture_output=True, encoding="utf-8",
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    (RUNS / f"{split}.log").write_text(" ".join(cmd[1:]) + "\n" + process.stdout + process.stderr,
                                     encoding="utf-8")
    process.check_returncode()
    return read_jsonl(target)


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


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    meta = load_meta(DATA / "metadata.json")
    meta.pop("trivial_baseline", None)
    dev = read_jsonl(DATA / "dev100.jsonl")
    inputs = RUNS / "dev.inputs.jsonl"
    write_rows(inputs, [{key: value for key, value in row.items() if key != "target"} for row in dev])
    predictions = run("dev", inputs)
    inputs.unlink()
    coverage = {"dev": assert_protocol(dev, predictions)}
    metrics, errors = evaluate(meta, dev, predictions)
    threshold = json.loads((RUNS / "training.json").read_text(encoding="utf-8"))["overlap_baseline"]
    baseline = []
    for row in dev:
        started = perf_counter()
        values = features(row["text_pair"])
        label = int(values[2] >= threshold["jaccard"] and values[3] >= threshold["containment"])
        baseline.append({"item_id": row["item_id"], "status": "ok", "output": {"label": label},
                         "latency_ms": int((perf_counter() - started) * 1000), "usage": USAGE})
    assert_protocol(dev, baseline)
    baseline_metrics, baseline_errors = evaluate(meta, dev, baseline)
    write_rows(RUNS / "results.baseline.dev.jsonl", baseline)
    write_rows(RUNS / "errors.dev.jsonl", errors)
    write_rows(RUNS / "errors.baseline.dev.jsonl", baseline_errors)
    for split, filename in (("test", "test_inputs.jsonl"), ("compare", "compare50.jsonl")):
        source = DATA / filename
        coverage[split] = assert_protocol(read_jsonl(source), run(split, source))
    write_json(RUNS / "metrics.json", {"trad": metrics, "baseline": baseline_metrics, "protocol": coverage})
    print(f"Dev accuracy: trad={metrics['accuracy']:.4f}, baseline={baseline_metrics['accuracy']:.4f}")
    if any(info["valid"] != info["rows"] for info in coverage.values()):
        raise SystemExit("Evaluation completed with invalid or failed outputs")


if __name__ == "__main__":
    main()
