"""Train on the course training set and select on the frozen dev1200 split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Resolve package-relative imports when invoked as python oj/<script>.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "oj"

from .model import FEATURE_NAMES, features, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = (16, 32, 64)
LEARNING_RATES = (0.003, 0.01)
CHECKPOINTS = (5, 10, 15, 20, 25, 30)
THRESHOLDS = tuple(round(i / 20, 2) for i in range(-12, 13))
BATCH_SIZE = 256
L2 = 0.0001


def metrics(np, scores, labels, threshold):
    predictions = scores >= threshold
    correct = int(np.count_nonzero(predictions == labels))
    f1 = []
    for label in (0, 1):
        expected = labels == label
        predicted = predictions == label
        denominator = int(expected.sum() + predicted.sum())
        f1.append(2 * int((expected & predicted).sum()) / denominator if denominator else 0.0)
    return {"correct": correct, "accuracy": correct / len(labels), "macro_f1": sum(f1) / 2}


def rank(trial):
    return (trial["correct"], trial["macro_f1"], -trial["width"], -trial["epoch"],
            -abs(trial["threshold"]))


def labeled(rows, np):
    values, labels = [], []
    for row in rows:
        try:
            label = row["target"]["label"]
            if type(label) is not int or label not in (0, 1):
                raise ValueError("invalid label")
            values.append(features(row["text_pair"]))
            labels.append(label)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{row.get('item_id')}: {error}") from error
    return np.asarray(values), np.asarray(labels)


def fit(np, training, tuning, width, learning_rate, seed):
    x, y = training
    vx, vy = tuning
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.15, (x.shape[1], width))
    b = np.zeros(width)
    u = rng.normal(0, 0.15, width)
    c = np.zeros(1)
    params = [w, b, u, c]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    step = 0

    for epoch in range(1, CHECKPOINTS[-1] + 1):
        indices = rng.permutation(len(y))
        loss = 0.0
        for start in range(0, len(y), BATCH_SIZE):
            batch = indices[start:start + BATCH_SIZE]
            inputs = x[batch]
            hidden = np.tanh(inputs @ w + b)
            scores = hidden @ u + c
            loss += np.sum(np.logaddexp(0, scores) - y[batch] * scores)
            delta = (1 / (1 + np.exp(-np.clip(scores, -50, 50))) - y[batch]) / len(batch)
            hidden_delta = (delta[:, None] * u) * (1 - hidden * hidden)
            gradients = [inputs.T @ hidden_delta + L2 * w, hidden_delta.sum(0),
                         hidden.T @ delta + L2 * u, np.array([delta.sum()])]
            step += 1
            for index, (param, gradient) in enumerate(zip(params, gradients)):
                first[index] = 0.9 * first[index] + 0.1 * gradient
                second[index] = 0.999 * second[index] + 0.001 * gradient * gradient
                corrected_first = first[index] / (1 - 0.9 ** step)
                corrected_second = second[index] / (1 - 0.999 ** step)
                param -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

        if epoch in CHECKPOINTS:
            scores = np.tanh(vx @ w + b) @ u + c
            thresholds = [{"threshold": t, **metrics(np, scores, vy, t)} for t in THRESHOLDS]
            best_threshold = max(
                thresholds, key=lambda t: (t["correct"], t["macro_f1"], -abs(t["threshold"]))
            )
            trial = {"width": width, "learning_rate": learning_rate, "epoch": epoch,
                     **best_threshold, "threshold_trials": thresholds,
                     "online_log_loss": float(loss / len(y))}
            model = {"hidden_weights": w.T.tolist(), "hidden_bias": b.tolist(),
                     "weights": u.tolist(), "bias": float(c[0]), "threshold": trial["threshold"]}
            yield trial, model


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.resolve() == ROOT or (args.output / "models/trad.json").exists():
        parser.error("--output must be a fresh experiment directory outside the submission root itself")
    try:
        import numpy as np
    except ImportError:
        parser.exit(2, "Training requires NumPy. Run: python -m pip install -r requirements.txt\n")

    sources = {"train.jsonl": 44898, "splits/dev1200.jsonl": 1200}
    freeze = json.loads((args.data / "BENCHMARK_FREEZE.json").read_text(encoding="utf-8"))
    for name in sources:
        if sha256(args.data / name) != freeze["sha256"][f"data/{name}"]:
            parser.error(f"frozen SHA-256 mismatch: {name}")
    train_rows, tuning_rows = (read_jsonl(args.data / name) for name in sources)
    if [len(train_rows), len(tuning_rows)] != list(sources.values()):
        parser.error("unexpected train/dev1200 size")
    print(f"seed={args.seed}, train={len(train_rows)}, tune={len(tuning_rows)}", flush=True)
    x, y = labeled(train_rows, np)
    vx, vy = labeled(tuning_rows, np)
    means, scales = x.mean(0), x.std(0)
    scales[scales == 0] = 1
    training = ((x - means) / scales, y)
    tuning = ((vx - means) / scales, vy)

    selected = None
    trials = []
    for width in WIDTHS:
        for learning_rate in LEARNING_RATES:
            for trial, model in fit(np, training, tuning, width, learning_rate, args.seed):
                trials.append(trial)
                print(f"width={width} lr={learning_rate:g} epoch={trial['epoch']} "
                      f"tuning={trial['correct']}/{len(vy)} macro_f1={trial['macro_f1']:.6f} "
                      f"threshold={trial['threshold']:g} "
                      f"log_loss={trial['online_log_loss']:.5f}", flush=True)
                if selected is None or rank(trial) > rank(selected):
                    selected, best = trial, model
    best.update({"kind": "tanh_mlp", "features": FEATURE_NAMES, "means": means.tolist(),
                 "scales": scales.tolist(), "seed": args.seed})
    model_path = args.output / "models/trad.json"
    write_json(model_path, best)
    summary = {key: value for key, value in selected.items() if key != "threshold_trials"}
    write_json(args.output / "runs/trad/training.json", {
        "train_rows": len(x), "tune_rows": len(vx), "seed": args.seed,
        "dataset_version": freeze["dataset_version"], "batch_size": BATCH_SIZE, "l2": L2,
        "source_sha256": {name: freeze["sha256"][f"data/{name}"] for name in sources},
        "model_sha256": sha256(model_path),
        "trials": trials, "selected": summary, "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    })
    print(f"selected={summary}", flush=True)


if __name__ == "__main__":
    main()
