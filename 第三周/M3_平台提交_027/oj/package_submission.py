"""Build the M3 submission from an explicit list of deliverables."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "M3实验报告.docx",
    "README.md",
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
    "runs/trad/dev.log",
    "runs/trad/errors.baseline.dev.jsonl",
    "runs/trad/errors.dev.jsonl",
    "runs/trad/metrics.json",
    "runs/trad/results.baseline.dev.jsonl",
    "runs/trad/results.compare.jsonl",
    "runs/trad/results.dev.jsonl",
    "runs/trad/results.test.jsonl",
    "runs/trad/selfcheck.log",
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
