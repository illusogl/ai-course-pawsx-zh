"""离线核验已保存的调用和预测，再按课程规则做配对比较。"""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
COURSE = REPO / "pawsx_zh/T4-PAR-中文改写语义一致性"
TRAD = REPO / "第四周/M4_平台提交_027"
RUNS = PACKAGE / "runs"
sys.path.insert(0, str(COURSE / "oj"))
from eval_split import LABELS, bootstrap_ci, gold_label, load_jsonl, pred_label, scoring


def digest(path):
    """计算完整文件的 SHA-256，核对实验使用的文件版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index(path, expected):
    """按题号索引记录，检查缺行、重复及多余题号。"""
    rows = load_jsonl(str(path))
    by_id = {row["item_id"]: row for row in rows}
    if len(rows) != len(by_id) or set(by_id) != set(expected):
        raise ValueError(f"{path.name} 的题号重复或覆盖范围不匹配")
    return by_id


def check_run(runs, model_config, variant, split, source, expected):
    """核验一次模型运行的配置、逐题调用次数和结果格式。"""
    path = runs / variant / f"{split}.jsonl"
    result = index(path, expected)
    calls = index(path.with_suffix(".calls.jsonl"), expected)
    info = json.loads(path.with_suffix(".run.json").read_text(encoding="utf-8"))
    model = json.loads(model_config.read_text(encoding="utf-8"))["request"]["model"]
    prompt = (PACKAGE / "prompts" / f"{variant}.txt").read_text(encoding="utf-8").strip()
    if (info["split"] != split or info["prompt"] != variant
            or info["item_count"] != len(expected) or info["input_sha256"] != digest(source)
            or info["system_message_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()
            or info["model_config_sha256"] != digest(model_config)):
        raise ValueError(f"{variant}/{split} 的运行配置与当前冻结内容不一致")
    for item_id, row in result.items():
        call = calls[item_id]
        if (set(row) != {"item_id", "output", "status", "latency_ms", "usage"}
                or row["status"] not in ("ok", "fail", "error", "timeout")
                or row["status"] != call["status"] or row["latency_ms"] != call["latency_ms"]
                or call["request_attempts"] != 1 or row["usage"]["tool_calls"] != 0
                or (row["output"] is not None) != (row["status"] == "ok")
                or (row["status"] == "ok" and pred_label(row) not in (0, 1))):
            raise ValueError(f"{variant}/{split}/{item_id} 的输出协议不匹配")
        if row["status"] == "ok":
            response = call["response"]
            message = response["choices"][0]["message"]
            if (call["http_status"] != 200 or response["model"] != model
                    or message.get("tool_calls") or json.loads(message["content"]) != row["output"]):
                raise ValueError(f"{variant}/{split}/{item_id} 的预测与原始响应不一致")
    return result, calls


def protocol(rows, calls=None):
    """统计输出覆盖、调用状态、服务端用量和逐题耗时。"""
    usage = [row["usage"] for row in rows.values()]
    latencies = sorted(row["latency_ms"] for row in rows.values())
    costs = [u["cost"] for u in usage if u["cost"] is not None]
    summary = {"rows": len(rows), "coverage": 1.0,
               "valid": sum(row["status"] == "ok" and pred_label(row) in (0, 1)
                            for row in rows.values()),
               "statuses": dict(Counter(row["status"] for row in rows.values())),
               "latency_ms_p50": statistics.median(latencies),
               "latency_ms_p95": statistics.quantiles(latencies, n=20, method="inclusive")[18],
               "tokens_in": sum(u["tokens_in"] for u in usage),
               "tokens_out": sum(u["tokens_out"] for u in usage),
               "known_cost_usd": sum(costs), "rows_with_cost": len(costs)}
    if calls is not None:
        responses = [call["response"] for call in calls.values()]
        billing = [r.get("usage", {}).get("opencode", {}) for r in responses if isinstance(r, dict)]
        summary.update(requests=sum(call["request_attempts"] for call in calls.values()),
                       allowance_usd=round(sum(u.get("allowance", 0) for u in billing), 8),
                       http_statuses=dict(Counter(str(call["http_status"]) for call in calls.values())))
    return summary


def compare(gold, methods):
    """复用课程判分器，按同一题号计算准确率差和各类错题。"""
    metrics = {}
    correct = {}
    errors = []
    for method, predictions in methods.items():
        records = [{"id": i, "prediction": row["output"] if row["status"] == "ok" else None}
                   for i, row in predictions.items()]
        score = scoring.score({"task": "single_label", "labels": LABELS},
                              {i: {"label": gold_label(row)} for i, row in gold.items()}, records)
        correct[method] = [int(pred_label(predictions[i]) == gold_label(row))
                           for i, row in gold.items()]
        score["accuracy_ci95"] = bootstrap_ci(correct[method])
        metrics[method] = score
    for i, row in gold.items():
        label = gold_label(row)
        outcomes = {}
        for method, predictions in methods.items():
            record = predictions[i]
            predicted = pred_label(record)
            if record["status"] != "ok":
                error = "format_invalid" if record["status"] == "fail" else record["status"]
            elif predicted != label:
                error = "false_positive" if predicted == 1 else "false_negative"
            else:
                error = None
            outcomes[method] = {"label": predicted, "status": record["status"], "error": error}
        if any(value["error"] for value in outcomes.values()):
            errors.append({"item_id": i, "gold": label, "text_pair": row["text_pair"],
                           "methods": outcomes})
    for method, score in metrics.items():
        score["error_counts"] = dict(Counter(row["methods"][method]["error"] for row in errors
                                             if row["methods"][method]["error"]))
    pairs = {}
    for left, right in (("zero-shot", "trad"), ("few-shot", "trad"), ("few-shot", "zero-shot")):
        delta = [a - b for a, b in zip(correct[left], correct[right])]
        pairs[f"{left}_minus_{right}"] = {
            "accuracy_delta": sum(delta) / len(delta), "ci95": bootstrap_ci(delta),
            "left_only_correct": delta.count(1), "right_only_correct": delta.count(-1),
        }
    return {"metrics": metrics, "paired": pairs}, errors


def write_jsonl(path, rows):
    """写出派生记录，每行对应一个题目。"""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    """从冻结数据和只读运行日志重算第五周结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="核验归档的 OpenCode Go 首轮记录")
    args = parser.parse_args()
    runs = RUNS / "opencode-go" if args.archive else RUNS
    config = runs / "model.json" if args.archive else PACKAGE / "model.json"
    data = COURSE / "data"
    benchmark = json.loads((TRAD / "benchmark.json").read_text(encoding="utf-8"))
    for relative, expected_hash in benchmark["data_sha256"].items():
        if digest(data / relative) != expected_hash:
            raise ValueError(f"冻结数据哈希不符：{relative}")
    for relative, expected_hash in benchmark["files"].items():
        if digest(TRAD / relative) != expected_hash:
            raise ValueError(f"M4 冻结材料哈希不符：{relative}")

    examples = json.loads(config.read_text(encoding="utf-8"))["prompt_examples"]["few-shot"]
    dev = {row["item_id"]: row for row in load_jsonl(str(data / "splits/dev1200.jsonl"))}
    prompt = (PACKAGE / "prompts/few-shot.txt").read_text(encoding="utf-8").strip()
    example_lines = prompt.split("示例：\n", 1)[1].splitlines()
    expected_examples = [entry for item_id in examples
                         for entry in ({"text_pair": dev[item_id]["text_pair"]}, dev[item_id]["target"])]
    if [json.loads(line) for line in example_lines] != expected_examples:
        raise ValueError("少样本提示与登记的开发集示例不一致")

    methods = {}
    summaries = {}
    holdout = data / "splits/holdout746.jsonl"
    gold = {row["item_id"]: row for row in load_jsonl(str(holdout))}
    for variant in ("zero-shot", "few-shot"):
        methods[variant], calls = check_run(runs, config, variant, "holdout746", holdout, gold)
        summaries[variant] = {"holdout746": protocol(methods[variant], calls)}
    trad = TRAD / "runs/trad"
    methods["trad"] = index(trad / "results.holdout746.jsonl", gold)
    summaries["trad"] = {"holdout746": protocol(methods["trad"])}
    comparison, errors = compare(gold, methods)
    frozen = json.loads((trad / "metrics.holdout.json").read_text(encoding="utf-8"))["holdout746"]
    if any(comparison["metrics"]["trad"][key] != frozen[key] for key in ("accuracy", "macro_f1")):
        raise ValueError("M4 留出预测与已冻结分数不一致")

    inputs = {}
    for split, name in (("test", "test_inputs"), ("dev100", "dev100"), ("compare50", "compare50")):
        source = data / f"{name}.jsonl"
        inputs[split] = {row["item_id"]: row for row in load_jsonl(str(source))}
        variant = "few-shot" if split == "compare50" else "zero-shot"
        rows, calls = check_run(runs, config, variant, split, source, inputs[split])
        summaries[variant][split] = protocol(rows, calls)
        if split == "test":
            test_rows, test_calls = rows, calls
        if split != "dev100":
            file_split = "compare" if split == "compare50" else split
            summaries["trad"][split] = protocol(index(trad / f"results.{file_split}.jsonl", inputs[split]))
    compare_ids = list(inputs["compare50"])
    if any(inputs["test"][i]["text_pair"] != inputs["compare50"][i]["text_pair"]
           for i in compare_ids):
        raise ValueError("compare50 与 test 的句对不一致")
    subset = {i: test_rows[i] for i in compare_ids}
    write_jsonl(runs / "zero-shot/compare50.jsonl", subset.values())
    summaries["zero-shot"]["compare50"] = protocol(subset, {i: test_calls[i] for i in compare_ids})
    summaries["zero-shot"]["compare50"]["derived_from"] = "zero-shot/test.jsonl"
    write_jsonl(runs / "errors.holdout746.jsonl", errors)
    summary = {"holdout746": comparison, "protocol": summaries}
    (runs / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
