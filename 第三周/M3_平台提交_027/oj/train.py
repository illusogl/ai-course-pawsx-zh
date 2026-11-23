"""Train a small tanh MLP and select its checkpoint on validation minus dev."""
import argparse
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
            correct, _, threshold = max(
                (int(((scores >= t) == vy).sum()), -abs(t), t) for t in THRESHOLDS
            )
            trial = {"width": width, "learning_rate": learning_rate, "epoch": epoch,
                     "tune_correct": correct, "threshold": threshold,
                     "online_log_loss": float(loss / len(y))}
            model = {"hidden_weights": w.T.tolist(), "hidden_bias": b.tolist(),
                     "weights": u.tolist(), "bias": float(c[0]), "threshold": threshold}
            yield trial, model


def choose_overlap_threshold(x, y):
    best = (-1, 0, 0)
    for jac in range(60, 100):
        for containment in range(80, 101):
            correct = int((((x[:, 2] >= jac / 100) & (x[:, 3] >= containment / 100)) == y).sum())
            if correct > best[0]:
                best = correct, jac, containment
    return {"correct": best[0], "jaccard": best[1] / 100, "containment": best[2] / 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError:
        parser.exit(2, "Training requires NumPy. Run: python -m pip install -r requirements.txt\n")

    train_rows = read_jsonl(args.data / "train.jsonl")
    validation = read_jsonl(args.data / "validation.jsonl")
    dev_ids = {row["item_id"] for row in read_jsonl(args.data / "dev100.jsonl")}
    tuning_rows = [row for row in validation if row["item_id"] not in dev_ids]
    excluded = len(validation) - len(tuning_rows)
    print(f"seed={args.seed}, train={len(train_rows)}, tune={len(tuning_rows)}, excluded_dev={excluded}", flush=True)
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
                      f"tuning={trial['tune_correct']}/{len(vy)} threshold={trial['threshold']:g} "
                      f"log_loss={trial['online_log_loss']:.5f}", flush=True)
                if selected is None or trial["tune_correct"] > selected["tune_correct"]:
                    selected, best = trial, model
    best.update({"kind": "tanh_mlp", "features": FEATURE_NAMES, "means": means.tolist(),
                 "scales": scales.tolist(), "seed": args.seed})
    write_json(ROOT / "models/trad.json", best)
    write_json(ROOT / "runs/trad/training.json", {
        "train_rows": len(x), "tune_rows": len(vx), "seed": args.seed,
        "dev_rows_excluded": excluded, "batch_size": BATCH_SIZE, "l2": L2,
        "trials": trials, "selected": selected, "numpy_version": np.__version__,
        "overlap_baseline": choose_overlap_threshold(vx, vy),
    })
    print(f"selected={selected}", flush=True)


if __name__ == "__main__":
    main()
