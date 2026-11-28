"""将模型与原生推理程序一并装入 M5 判题包。"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "pawsx_zh/T4-PAR-中文改写语义一致性/data"
FILES = ("M5实验报告.docx", "README.md", "manifest.json", "model.json", "platform.json",
         "oj/run.py", "oj/local_model.py", "oj/evaluate.py", "prompts/zero-shot.txt",
         "prompts/few-shot.txt", "licenses/Qwen-Apache-2.0.txt")
LINUX = {"llama-server", "libllama-server-impl.so", "libllama-common.so.1", "libllama.so.1",
         "libmtmd.so.1", "libggml.so.1", "libggml-base.so.1", "LICENSE"}
WINDOWS = {"llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "llama.dll",
           "mtmd.dll", "ggml.dll", "ggml-base.dll", "libomp.dll", "LICENSE-LLVM-OpenMP"}


def verify_asset(path, expected):
    """核对下载的模型或运行时文件，拒绝版本和哈希不符的文件。"""
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(block)
    if actual.hexdigest() != expected:
        raise ValueError(f"SHA-256 不匹配：{path.name}")


def add_runtime(archive, source, target):
    """挑出平台需要的模型程序和动态库，并保留随发布包的许可。"""
    if target == "linux":
        with tarfile.open(source, "r:gz") as runtime:
            members = {Path(m.name).name: m for m in runtime.getmembers() if not m.isdir()}
            selected = LINUX | {name for name in members if name.startswith("libggml-cpu-")}
            for name in sorted(selected):
                # 将 SONAME 链接展开为文件，兼容不保留符号链接的解包器。
                with runtime.extractfile(members[name]) as stream:
                    archive.writestr(f"runtime/linux/{name}", stream.read())
    else:
        with ZipFile(source) as runtime:
            names = runtime.namelist()
            selected = WINDOWS | {name for name in names if name.startswith("ggml-cpu-")}
            for name in sorted(selected):
                archive.writestr(f"runtime/windows/{name}", runtime.read(name))


def main():
    """生成自包含压缩包，并在替换目标文件前核验大小及完整性。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="官方权重和运行时压缩包目录")
    parser.add_argument("--target", choices=["linux", "windows"], default="linux")
    parser.add_argument("--out", type=Path, default=ROOT / "package.zip")
    args = parser.parse_args()
    settings = json.loads((ROOT / "platform.json").read_text(encoding="utf-8"))
    weights = args.assets / settings["model"]["file"]
    distribution = settings["runtime"][args.target]
    runtime = args.assets / distribution["archive"]
    verify_asset(weights, settings["model"]["sha256"])
    verify_asset(runtime, distribution["sha256"])
    # Windows 构建也随包附带 llama.cpp 许可，发行版副本来自固定的 Linux 发布包。
    license_source = args.assets / settings["runtime"]["linux"]["archive"]
    if args.target == "windows":
        verify_asset(license_source, settings["runtime"]["linux"]["sha256"])
    target = args.out
    temporary = target.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=1) as archive:
            for name in FILES:
                archive.write(ROOT / name, name)
            for path in sorted((ROOT / "runs").rglob("*")):
                if path.suffix in (".json", ".jsonl"):
                    archive.write(path, path.relative_to(ROOT))
            for name in ("metadata.json", "dev100.jsonl", "compare50.jsonl", "test_inputs.jsonl"):
                archive.write(DATA / name, "data/" + name)
            archive.write(weights, "models/" + weights.name)
            add_runtime(archive, runtime, args.target)
            if args.target == "windows":
                with tarfile.open(license_source, "r:gz") as release:
                    with release.extractfile(f"llama-{settings['runtime']['release']}/LICENSE") as stream:
                        archive.writestr("runtime/windows/LICENSE", stream.read())
        if temporary.stat().st_size > 450_000_000:
            raise ValueError("判题包超过 450 MB")
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("压缩包校验失败")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{target.name}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
