"""打包第五周入口、课程小样本和实际运行材料。"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[1] / "pawsx_zh/T4-PAR-中文改写语义一致性/data"
FILES = ("M5实验报告.docx", "README.md", "manifest.json", "model.json", "requirements.txt",
         "oj/run.py", "oj/evaluate.py", "prompts/zero-shot.txt", "prompts/few-shot.txt")


def main():
    """生成压缩包，并在替换目标文件前确认包内材料可读。"""
    target = ROOT / "package.zip"
    temporary = target.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(ROOT / name, name)
            for path in sorted((ROOT / "runs").rglob("*")):
                if path.suffix in (".json", ".jsonl"):
                    archive.write(path, path.relative_to(ROOT))
            for name in ("metadata.json", "dev100.jsonl", "compare50.jsonl", "test_inputs.jsonl"):
                archive.write(DATA / name, "data/" + name)
        with ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError("压缩包校验失败")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{target.name}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
