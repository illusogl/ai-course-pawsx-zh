"""逐题调用单一模型，写出课程结果与原始调用记录。"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from local_model import start_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "model.json"
PLATFORM = ROOT / "platform.json"
LOCAL = threading.local()


def external_request(config, session_id, key):
    """保留历史 HTTP 实验的调用入口。"""
    import requests

    headers = {"User-Agent": "pawsx-zh-m5/1.0"}
    if key:
        headers.update({"Authorization": f"Bearer {key}", "x-opencode-session": session_id})

    def request(payload):
        """向指定的实验服务发送一次请求。"""
        if not hasattr(LOCAL, "session"):
            LOCAL.session = requests.Session()
        try:
            with LOCAL.session.post(config["endpoint"], json=payload, headers=headers,
                                    timeout=config["timeout_seconds"], allow_redirects=False) as reply:
                body = reply.text.replace(key, "[REDACTED]") if key else reply.text
                return reply.status_code, body
        except requests.Timeout as error:
            raise TimeoutError from error
        except requests.RequestException as error:
            raise OSError(type(error).__name__) from error

    return request


def classify(item_id, pair, prompt, config, request):
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
        event["http_status"], body = request(payload)
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
    except TimeoutError:
        result["status"] = "timeout"
        event["reason"] = "request timed out"
    except (OSError, http.client.HTTPException) as error:
        event["reason"] = f"{type(error).__name__}: {error}"
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
        item_id = row.get("item_id", row.get("id"))
        pair = row.get("text_pair") if "text_pair" in row else row.get("input", {}).get("text_pair")
        if (not isinstance(item_id, str) or not item_id or item_id in items
                or not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(sentence, str) for sentence in pair)):
            raise ValueError(f"无效句对或重复 ID {item_id!r}")
        items[item_id] = pair
    if not items:
        raise ValueError("输入没有有效题目")
    return items.items()


def write_results(args, config, prompt, items, run_info, engine, audit, metadata):
    """启动模型并写出逐题结果，同时记录未经改写的模型响应。"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with (engine as request,
          metadata.open("x", encoding="utf-8") as meta,
          audit.open("x", encoding="utf-8") as call_log,
          args.out.open("a", encoding="utf-8") as output,
          ThreadPoolExecutor(max_workers=args.workers) as pool):
        json.dump(run_info, meta, ensure_ascii=False, indent=2)
        meta.write("\n")
        meta.flush()
        futures = [pool.submit(classify, item_id, pair, prompt, config, request)
                   for item_id, pair in items]
        for index, future in enumerate(futures, 1):
            result, event = future.result()
            call_log.write(json.dumps(event, ensure_ascii=False) + "\n")
            call_log.flush()
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            if index % 100 == 0 or index == len(futures):
                print(f"{args.split} {args.prompt}: {index}/{len(futures)}", flush=True)


def main():
    """按课程参数运行，保留每题的结果、状态和调用记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["llm"], default="llm")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--config", type=Path, help="改用历史实验的 HTTP 模型配置")
    parser.add_argument("--local-port", type=int, help="改用已启动的本地 llama.cpp 服务")
    args = parser.parse_args()
    config_path = args.config or CONFIG
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundled = args.config is None and args.local_port is None
    settings = json.loads(PLATFORM.read_text(encoding="utf-8")) if bundled else None
    config["request"]["seed"] = args.seed
    if bundled:
        config["request"]["model"] = settings["model"]["alias"]
        config["endpoint"] = "bundled"
    if args.workers is None:
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        args.workers = min(settings["max_workers"], cores) if bundled else 1
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
    if (any(p.exists() or p.is_symlink() for p in (audit, metadata))
            or args.out.is_symlink() or (args.out.exists() and args.out.stat().st_size)):
        parser.error("运行产物已存在，请另选输出路径")

    try:
        items = read_items(args.input)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.error(f"无法读取输入：{error}")
    prompt_path = ROOT / "prompts" / f"{args.prompt}.txt"
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    session_id = str(uuid4())
    run_info = {
        "split": args.split, "input_file": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "prompt": args.prompt, "system_message_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "seed": args.seed,
        "endpoint": config["endpoint"],
        "session_id": session_id, "workers": args.workers, "item_count": len(items),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if bundled:
        run_info["platform_config_sha256"] = hashlib.sha256(PLATFORM.read_bytes()).hexdigest()
        engine = start_model(ROOT, settings, args.workers)
    else:
        engine = nullcontext(external_request(config, session_id, key))
    try:
        write_results(args, config, prompt, items, run_info, engine, audit, metadata)
    except (OSError, RuntimeError) as error:
        print(f"M5 {args.split}: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
