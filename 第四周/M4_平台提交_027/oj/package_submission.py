"""Build the M4 submission from an explicit list of deliverables."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "M4实验报告.docx",
    "README.md",
    "调参记录.md",
    "误差分析.md",
    "benchmark.json",
    "manifest.json",
    "models/trad.json",
    "oj/__init__.py",
    "oj/evaluate.py",
    "oj/model.py",
    "oj/package_submission.py",
    "oj/run.py",
    "oj/scoring.py",
    "oj/selfcheck.py",
    "oj/train.py",
    "requirements.txt",
    "runs/trad/compare.log",
    "runs/trad/dev100.log",
    "runs/trad/dev1200.a.log",
    "runs/trad/dev1200.b.log",
    "runs/trad/holdout746.log",
    "runs/trad/errors.dev1200.jsonl",
    "runs/trad/metrics.json",
    "runs/trad/metrics.holdout.json",
    "runs/trad/results.compare.jsonl",
    "runs/trad/results.dev100.jsonl",
    "runs/trad/results.dev1200.a.jsonl",
    "runs/trad/results.dev1200.b.jsonl",
    "runs/trad/results.holdout746.jsonl",
    "runs/trad/results.test.jsonl",
    "runs/trad/test.log",
    "runs/trad/train.log",
    "runs/trad/training.json",
)


def main():
    target = ROOT / "package.zip"
    temporary = target.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(ROOT / name, name)
        if temporary.stat().st_size > 450_000_000:
            raise ValueError("Submission exceeds 450 MB")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{target.name}: {target.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
