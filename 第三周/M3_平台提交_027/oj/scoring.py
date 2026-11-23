"""确定性判分核心：单标签 / 多标签 / 多方面（口径与教师包 `course_pack/evaluate.py` 完全一致）。

为什么单独成文件：`runner.py` 原来只做「output 字符串 == gold.label 字符串」的 exact match，
对 GoEmotions（多标签）和 ASAP（18 方面）会给出 accuracy=0 且仍然判 pass（实测）。
本模块把教师包 evalauate.py 的口径 1:1 搬过来，供 runner 调用，也让本地自检与平台判题同源。

口径（照抄教师包，**不要**"改良"）：
  1. 缺行、重复 ID、未知 ID、非法标签**全部留在分母里**（分母恒为 gold 条数）；
  2. 重复 ID 视为无效（不是"最后一条覆盖前面"）；
  3. 类别集合固定取 meta 的 labels / aspects / allowed_labels；没有样本的类别按 0 计入宏平均；
  4. ASAP 先按方面算 allowed_labels（默认 4 值）上的宏 F1，再对 18 个方面取平均；
  5. 非法预测值用哨兵 -999 参与计算，保证"答了但答错"与"没答"都不被美化。

任务形态由 meta["task"] 决定：single_label / multilabel / multi_aspect。
"""
from __future__ import annotations

import json

TASKS = ("single_label", "multilabel", "multi_aspect")
SENTINEL = -999


# ------------------------------------------------------------------ 读取
def _rows(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_items(path: str) -> list[dict]:
    """items.jsonl：每行至少 `item_id`；可带 `input` / `task`。"""
    return _rows(path)


def load_gold(path: str) -> dict[str, dict]:
    """gold.jsonl：兼容三种写法，统一成 {item_id: target_dict}。

    - 课程标准：`{"item_id": "...", "target": {"label": 3}}`（教师包 teacher_only 的风格，id/target）
    - 旧占位：  `{"item_id": "...", "label": "pos"}`（P1c 合成数据）
    - 也接受 `id` 作为主键字段名。
    """
    gold: dict[str, dict] = {}
    for r in _rows(path):
        iid = r.get("item_id") or r.get("id")
        if not iid:
            continue
        if isinstance(r.get("target"), dict):
            gold[iid] = r["target"]
        elif "label" in r:
            gold[iid] = {"label": r["label"]}
        elif "labels" in r:
            gold[iid] = {"labels": r["labels"]}
        elif "aspects" in r:
            gold[iid] = {"aspects": r["aspects"]}
        else:
            gold[iid] = {}
    return gold


def load_meta(path: str) -> dict:
    """meta.json：数据集元信息（task / labels / aspects / allowed_labels / oos_label）。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def meta_path_for(items_path: str) -> str | None:
    """benchmark 目录约定：items.*.jsonl 同目录下的 meta.json。"""
    import os

    d = os.path.dirname(os.path.abspath(items_path))
    for name in ("meta.json", "task.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


# ------------------------------------------------------------------ 预测值归一
def parse_output(raw):
    """把 results.jsonl 的 `output` 变成结构；字符串里带 JSON 就解析，否则原样返回。"""
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    if not s:
        return None
    if s[0] in "{[":
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return s


def _as_int(x) -> int | None:
    """严格整数：bool 不算整数（Python 里 True == 1，必须排除）。"""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        try:
            return int(x.strip())
        except ValueError:
            return None
    return None


def _label_of(pred) -> int | None:
    if isinstance(pred, dict):
        return _as_int(pred.get("label"))
    return _as_int(pred)


def _label_name_like(pred, labels) -> bool:
    """学生误把标签名（如 accept_reservations）当答案：判无效，但要给出可行动的提示。"""
    name = pred.get("label") if isinstance(pred, dict) else pred
    return isinstance(name, str) and name.strip() in labels


def _labels_of(pred) -> list[int] | None:
    raw = pred.get("labels") if isinstance(pred, dict) else pred
    if not isinstance(raw, list):
        return None
    vals = [_as_int(v) for v in raw]
    if any(v is None for v in vals):
        return None
    return vals  # type: ignore[return-value]


def _aspects_of(pred) -> dict | None:
    """兼容两种写法：`{"aspects": {...}}`（教师包文件格式）与裸 `{方面: 值, ...}`。"""
    if not isinstance(pred, dict):
        return None
    if isinstance(pred.get("aspects"), dict):
        return pred["aspects"]
    if pred and all(isinstance(k, str) for k in pred):
        return pred
    return None


# ------------------------------------------------------------------ 指标
def _macro_f1(y: list, p: list, classes) -> float:
    scores = []
    for c in classes:
        tp = sum(a == c and b == c for a, b in zip(y, p))
        fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score(meta: dict, gold: dict[str, dict], predictions: list[dict]) -> dict:
    """对一批预测打分。predictions 为 [{"id": ..., "prediction": ...}, ...]（重复 ID 会被判无效）。

    返回 metrics；其中 valid_output_rate 是"格式合法率"，不合格的行不缩小分母。
    """
    task = meta.get("task")
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task!r}（期望 {TASKS}）")
    if not gold:
        raise ValueError("empty gold")

    gold_ids = list(gold.keys())
    gold_set = set(gold_ids)
    n = len(gold_ids)

    accepted: dict[str, object] = {}
    duplicates: set[str] = set()
    unknown = 0
    for r in predictions:
        if not isinstance(r, dict) or (r.get("id") or r.get("item_id")) not in gold_set:
            unknown += 1
            continue
        iid = r.get("id") or r.get("item_id")
        if iid in accepted:
            duplicates.add(iid)
        accepted[iid] = r.get("prediction", r.get("output"))

    for iid in duplicates:          # 重复 = 无效，不是"最后一条生效"
        accepted[iid] = None

    missing = sum(1 for iid in gold_ids if iid not in accepted)
    valid = 0
    result: dict = {
        "total": n,
        "prediction_records": len(predictions),
        "missing": missing,
        "duplicate_ids": len(duplicates),
        "unknown_records": unknown,
    }

    if task == "single_label":
        labels = meta["labels"]
        classes = range(len(labels))
        y, p = [], []
        name_like = 0
        for iid in gold_ids:
            g = _as_int(gold[iid].get("label"))
            raw = parse_output(accepted.get(iid))
            v = _label_of(raw)
            ok = v is not None and v in classes
            valid += ok
            name_like += _label_name_like(raw, labels)
            y.append(g)
            p.append(v if ok else SENTINEL)
        result["accuracy"] = sum(a == b for a, b in zip(y, p)) / n
        result["macro_f1"] = _macro_f1(y, p, classes)
        if "oos_label" in meta:
            c = meta["oos_label"]
            tp = sum(a == c and b == c for a, b in zip(y, p))
            fp = sum(a != c and b == c for a, b in zip(y, p))
            fn = sum(a == c and b != c for a, b in zip(y, p))
            result["oos_precision"] = tp / (tp + fp) if tp + fp else 0.0
            result["oos_recall"] = tp / (tp + fn) if tp + fn else 0.0
            result["oos_f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            # 正常类（非 OOS）上的 F1：识别"范围外问题"时，别把正常请求全推给人工
            in_classes = [c2 for c2 in classes if c2 != c]
            result["in_domain_macro_f1"] = _macro_f1(y, p, in_classes)

    elif task == "multilabel":
        size = len(meta["labels"])
        counts = [[0, 0, 0] for _ in range(size)]
        exact = 0
        label_card = 0
        for iid in gold_ids:
            g = set(_as_int(x) for x in gold[iid].get("labels", []))
            vals = _labels_of(parse_output(accepted.get(iid)))
            ok = (vals is not None and len(vals) > 0
                  and all(0 <= x < size for x in vals) and len(vals) == len(set(vals)))
            valid += ok
            pred = set(vals) if ok else set()
            label_card += len(pred)
            exact += ok and pred == g
            for c in range(size):
                counts[c][0] += c in g and c in pred
                counts[c][1] += c not in g and c in pred
                counts[c][2] += c in g and c not in pred
        tp = sum(c[0] for c in counts)
        fp = sum(c[1] for c in counts)
        fn = sum(c[2] for c in counts)
        result.update({
            "exact_match": exact / n,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "macro_f1": sum(2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
                            for a, b, c in counts) / size,
            "mean_labels_per_item": label_card / n,
            "gold_mean_labels_per_item": sum(len(gold[i]["labels"]) for i in gold_ids) / n,
        })

    else:  # multi_aspect
        aspects = meta["aspects"]
        values = meta.get("allowed_labels", [-2, -1, 0, 1])
        ys = {a: [] for a in aspects}
        ps = {a: [] for a in aspects}
        exact = 0
        correct = 0
        for iid in gold_ids:
            pred = _aspects_of(parse_output(accepted.get(iid)))
            ok = (pred is not None and set(pred) == set(aspects)
                  and all((_as_int(x) is not None and _as_int(x) in values) for x in pred.values()))
            valid += ok
            all_correct = True
            for a in aspects:
                g = _as_int(gold[iid]["aspects"][a])
                v = _as_int(pred.get(a)) if isinstance(pred, dict) else None
                if v is None or v not in values:
                    v = SENTINEL
                ys[a].append(g)
                ps[a].append(v)
                correct += g == v
                all_correct &= g == v
            exact += ok and all_correct
        result.update({
            "aspect_accuracy": correct / (n * len(aspects)),
            "macro_f1": sum(_macro_f1(ys[a], ps[a], values) for a in aspects) / len(aspects),
            "exact_match": exact / n,
        })

    result.update(valid_output_count=int(valid), valid_output_rate=valid / n)
    result["task"] = task
    if task == "single_label" and name_like:
        result["name_output_records"] = name_like
        result["hint"] = ('检测到标签名而非整数下标：单标签预测要求 {"label": 整数下标}'
                          '（下标取自 metadata.json 的 labels 顺序）')
    return result


# ------------------------------------------------------------------ 便捷入口（给 runner 用）
def score_rows(meta: dict, gold: dict[str, dict], rows: list[dict]) -> dict:
    """把 runner 的 results.jsonl 行（item_id / output）转成 predictions 再打分。"""
    preds = [{"id": r.get("item_id"), "prediction": r.get("output")} for r in rows]
    return score(meta, gold, preds)


def trivial_baselines(meta: dict, gold: dict[str, dict]) -> dict:
    """常数/退化基线分数：验收下界必须**高于**这些数，否则门禁形同虚设。

    这是实测教训：ASAP 全预测 -2 能拿到 aspect_accuracy≈0.67，但 macro_f1 只有≈0.20。
    """
    out: dict[str, dict] = {}
    if meta.get("task") == "single_label":
        from collections import Counter
        c = Counter(_as_int(g.get("label")) for g in gold.values())
        if c:
            top = c.most_common(1)[0][0]
            out["majority_label"] = score(meta, gold, [{"id": i, "prediction": top} for i in gold])
    elif meta.get("task") == "multilabel":
        from collections import Counter
        c = Counter(x for g in gold.values() for x in g.get("labels", []))
        if c:
            out["most_frequent_single_label"] = score(
                meta, gold, [{"id": i, "prediction": [c.most_common(1)[0][0]]} for i in gold])
    elif meta.get("task") == "multi_aspect":
        low = meta.get("allowed_labels", [-2, -1, 0, 1])[0]
        out["all_lowest_label"] = score(
            meta, gold,
            [{"id": i, "prediction": {"aspects": {a: low for a in meta["aspects"]}}} for i in gold])
    return out
