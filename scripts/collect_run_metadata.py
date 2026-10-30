"""Collect missing small run metadata locally. No model loading, network or GPU."""

import argparse, json, zipfile
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
a = p.parse_args()
if a.output.exists():
    p.error("Output already exists; choose a new filename")
names = [
    "config.json",
    "trainer_config.json",
    "data_manifest.json",
    "model_revision.json",
    "environment.txt",
    "train.jsonl",
    "dev.jsonl",
]
missing = [n for n in names if not (a.run / n).is_file()]
if missing:
    p.error("Missing metadata: " + ", ".join(missing))
if sum((a.run / n).stat().st_size for n in names) > 20 * 1024 * 1024:
    p.error("Unexpected metadata size; inspect before packaging")
with zipfile.ZipFile(a.output, "w", zipfile.ZIP_DEFLATED) as z:
    for name in names:
        z.write(a.run / name, "run_metadata/" + name)
print("Created:", a.output)
