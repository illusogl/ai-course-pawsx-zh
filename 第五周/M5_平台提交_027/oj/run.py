"""逐题调用单一模型，写出课程结果与原始调用记录。"""

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import requests


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "model.json"
LOCAL = threading.local()


def classify(item_id, pair, prompt, config, headers, key):
    """发送一次请求，返回平台结果以及未经改写的服务端答复。"""
    started = perf_counter()
    user_message = json.dumps({"text_pair": pair}, ensure_ascii=False)
    payload = {**config["request"], "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_message},
    ]}
    result = {"item_id": item_id, "output": None, "status": "error",
              "latency_ms": 0, "usage": {
                  "tokens_in": 0, "tokens_out": 0,
                  "cost": None if config.get("api_key_required", True) else 0, "tool_calls": 0,
              }}
    event = {"item_id": item_id, "started_at": datetime.now(timezone.utc).isoformat(),
             "request_attempts": 1, "http_status": None, "response": None}
    try:
        # 每个线程复用连接。Requests 默认关闭重试，也明确禁用 HTTP 跳转。
        if not hasattr(LOCAL, "session"):
            LOCAL.session = requests.Session()
        with LOCAL.session.post(config["endpoint"], json=payload, headers=headers,
                                timeout=config["timeout_seconds"], allow_redirects=False) as reply:
            event["http_status"] = reply.status_code
            body = reply.text.replace(key, "[REDACTED]") if key else reply.text
        try:
            response = json.loads(body)
        except ValueError:
            response = body
        event["response"] = response
        if event["http_status"] != 200:
            event["reason"] = f"HTTP {event['http_status']}"
        else:
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            result["usage"].update(tokens_in=usage.get("prompt_tokens", 0),
                                   tokens_out=usage.get("completion_tokens", 0),
                                   cost=usage.get("cost", result["usage"]["cost"]))
            try:
                choice, = response["choices"]
                message = choice["message"]
                if choice["finish_reason"] != "stop" or message.get("tool_calls"):
                    raise ValueError("unexpected finish reason or tool output")
                answer = json.loads(message["content"], object_pairs_hook=list)
                if (not isinstance(answer, list) or len(answer) != 1
                        or not isinstance(answer[0], tuple) or answer[0][0] != "label"
                        or type(answer[0][1]) is not int or answer[0][1] not in (0, 1)):
                    raise ValueError("expected one integer label")
                result.update(output={"label": answer[0][1]}, status="ok")
            except (KeyError, IndexError, ValueError, TypeError, AttributeError) as error:
                result["status"] = "fail"
                event["reason"] = f"invalid model answer: {type(error).__name__}"
    except requests.Timeout:
        result["status"] = "timeout"
        event["reason"] = "request timed out"
    except requests.RequestException as error:
        event["reason"] = type(error).__name__
    result["latency_ms"] = round((perf_counter() - started) * 1000, 1)
    event.update(status=result["status"], latency_ms=result["latency_ms"])
    return result, event


def read_items(path):
    """检查 ID 和句对，构造独立的模型输入，不传递标签或题号。"""
    items = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        item_id, pair = row["item_id"], row["text_pair"]
        if (not isinstance(item_id, str) or not item_id or item_id in items
                or not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(sentence, str) for sentence in pair)):
            raise ValueError(f"无效句对或重复 ID {item_id!r}")
        items[item_id] = pair
    if not items:
        raise ValueError("输入没有有效题目")
    return items.items()


def main():
    """按课程参数运行，保留每题的结果、状态和调用记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["llm"], default="llm")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--prompt", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--local-port", type=int, help="使用另一个本地 llama.cpp 监听端口")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.local_port is not None:
        if config.get("api_key_required", True) or not 1 <= args.local_port <= 65535:
            parser.error("--local-port 仅支持本地模型和有效端口号")
        config["endpoint"] = f"http://127.0.0.1:{args.local_port}/v1/chat/completions"
    key = os.environ.get("M5_API_KEY") if config.get("api_key_required", True) else None
    if config.get("api_key_required", True) and not key:
        parser.error("请设置 M5_API_KEY")
    if args.workers < 1:
        parser.error("--workers 必须是正整数")
    audit = args.out.with_suffix(".calls.jsonl")
    metadata = args.out.with_suffix(".run.json")
    if len({p.resolve() for p in (args.input, args.out, audit, metadata)}) != 4:
        parser.error("输入、输出、调用记录和配置必须使用不同路径")
    if any(p.exists() for p in (args.out, audit, metadata)):
        parser.error("运行产物已存在，请另选输出路径")

    try:
        items = read_items(args.input)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"无法读取输入：{error}")
    prompt_path = ROOT / "prompts" / f"{args.prompt}.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    session_id = str(uuid4())
    headers = {"User-Agent": "pawsx-zh-m5/1.0"}
    if key:
        headers.update({"Authorization": f"Bearer {key}", "x-opencode-session": session_id})
    run_info = {
        "split": args.split, "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "prompt": args.prompt, "system_message_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "requests_version": requests.__version__, "python": sys.version.split()[0],
        "endpoint": config["endpoint"],
        "session_id": session_id, "workers": args.workers, "item_count": len(items),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (metadata.open("x", encoding="utf-8") as meta,
          args.out.open("x", encoding="utf-8") as output,
          audit.open("x", encoding="utf-8") as call_log,
          ThreadPoolExecutor(max_workers=args.workers) as pool):
        json.dump(run_info, meta, ensure_ascii=False, indent=2)
        meta.write("\n")
        meta.flush()
        futures = [pool.submit(classify, item_id, pair, prompt, config, headers, key)
                   for item_id, pair in items]
        for index, future in enumerate(futures, 1):
            result, event = future.result()
            call_log.write(json.dumps(event, ensure_ascii=False) + "\n")
            call_log.flush()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if index % 100 == 0 or index == len(futures):
                print(f"{args.split} {args.prompt}: {index}/{len(futures)}", flush=True)


if __name__ == "__main__":
    main()
