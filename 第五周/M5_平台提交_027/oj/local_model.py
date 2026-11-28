"""启动和关闭随判题包提供的本地模型。"""

import http.client
import json
import os
import platform
import shutil
import socket
import subprocess
import sysconfig
import tempfile
import time
from contextlib import closing, contextmanager
from functools import partial
from pathlib import Path


class UnixHTTPConnection(http.client.HTTPConnection):
    """使用 Unix 文件套接字连接随包模型进程。"""

    def __init__(self, address, timeout):
        """保存套接字文件地址和 HTTP 超时时间。"""
        super().__init__("localhost", timeout=timeout)
        self.address = address

    def connect(self):
        """建立本地文件套接字连接，不使用 IP 网络。"""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.settimeout(self.timeout)
            client.connect(str(self.address))
        except OSError:
            client.close()
            raise
        self.sock = client


def http_call(connect, method, path, timeout, payload=None):
    """执行一次本地 HTTP 请求并读取状态码与响应正文。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    with closing(connect(timeout=timeout)) as client:
        client.request(method, path, body=body, headers=headers)
        with client.getresponse() as reply:
            return reply.status, reply.read().decode("utf-8", errors="replace")


def wait_until_ready(process, connect, timeout, stderr):
    """等待模型加载，启动失败时附上模型进程的错误日志。"""
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        try:
            status, _ = http_call(connect, "GET", "/health", timeout=1)
            if status == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(0.2)
    stderr.seek(0, os.SEEK_END)
    stderr.seek(max(0, stderr.tell() - 3000))
    tail = stderr.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"内置模型启动失败 (exit={process.poll()})\n{tail}")


@contextmanager
def start_model(root, settings, workers):
    """校验随包文件并启动一次模型进程，返回逐题调用函数。"""
    system = platform.system()
    architecture = sysconfig.get_platform().rsplit("-", 1)[-1].lower()
    if system not in ("Linux", "Windows") or architecture not in ("x86_64", "amd64"):
        raise RuntimeError(f"内置推理程序不支持 {system} {architecture}")
    runtime = root / "runtime" / system.lower()
    model = root / "models" / settings["model"]["file"]
    executable = "llama-server.exe" if system == "Windows" else "llama-server"
    if not (runtime / executable).is_file() or not model.is_file():
        raise RuntimeError("缺少内置推理程序或模型权重，请使用完整的 M5 判题包")

    # 服务从临时目录运行，平台解压时即使丢失可执行权限也能正常启动。
    with tempfile.TemporaryDirectory(prefix="m5-", dir="/tmp" if system == "Linux" else None) as folder:
        temporary = Path(folder)
        binary = temporary / "bin"
        shutil.copytree(runtime, binary)
        server = binary / executable
        server.chmod(0o755)
        env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
        if system == "Linux":
            host = str(temporary / "model.sock")
            port = 8080
            connect = partial(UnixHTTPConnection, host)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                part for part in (str(binary), env.get("LD_LIBRARY_PATH")) if part
            )
        else:
            host = "127.0.0.1"
            with closing(socket.socket()) as reservation:
                reservation.bind((host, 0))
                port = reservation.getsockname()[1]
            connect = partial(http.client.HTTPConnection, host, port)
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        threads = min(8, cores)
        command = [str(server), "-m", str(model), "--alias", settings["model"]["alias"],
                   "--host", host, "--port", str(port), "-ngl", "0",
                   "-c", str(workers * 2048), "-np", str(workers),
                   "-t", str(threads), "-tb", str(threads), "--no-webui", "--offline"]
        with (temporary / "server.log").open("w+b") as stderr:
            process = subprocess.Popen(command, cwd=binary, env=env, stderr=stderr,
                                       stdout=stderr, stdin=subprocess.DEVNULL,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                wait_until_ready(process, connect, settings["startup_timeout_seconds"], stderr)
                yield partial(http_call, connect, "POST", "/v1/chat/completions",
                              settings["request_timeout_seconds"])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
