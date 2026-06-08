#!/usr/bin/env python3
"""Make sure the challenge dataset is present in ./dataset/.

The synthetic dataset is committed to this repo, so a fresh clone already has
``dataset/train.csv``, ``dataset/validation.csv``, and ``dataset/test.csv``.
Running this script is then a no-op (it just confirms the files are there).

It is kept as a convenience for the one case where only the compressed archive
is present: drop ``dataset-compressed.zip`` into ``dataset/`` and run

  python scripts/setup_data.py

and it extracts the three CSVs.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
ZIP_PATH = DATASET_DIR / "dataset-compressed.zip"
EXPECTED = ("train.csv", "validation.csv", "test.csv")
OFFICIAL = "https://github.com/intuit/intuit-techweek-nyc-hackathon-2026"


def main() -> int:
    if all((DATASET_DIR / f).exists() for f in EXPECTED):
        print("Data already present:", ", ".join(EXPECTED))
        return 0

    if not ZIP_PATH.exists():
        print(
            f"ERROR: {ZIP_PATH} not found.\n\n"
            f"Download dataset-compressed.zip from the official repo:\n"
            f"  {OFFICIAL}\n"
            f"and place it in {DATASET_DIR}, then re-run this script.",
            file=sys.stderr,
        )
        return 1

    print(f"Extracting {ZIP_PATH.name} -> {DATASET_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATASET_DIR)

    missing = [f for f in EXPECTED if not (DATASET_DIR / f).exists()]
    if missing:
        print(f"WARNING: still missing after unzip: {missing}", file=sys.stderr)
        return 1
    print("Done. Extracted:", ", ".join(EXPECTED))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
