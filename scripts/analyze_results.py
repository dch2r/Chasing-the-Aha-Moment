"""Analyze recorded paired outcomes and export changed-answer review evidence."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_scoring import score
from paired_analysis import paired_outcomes


def analyze():
    protocol = json.loads((ROOT / "results/evaluation/protocol.json").read_text())
    rows = [
        json.loads(line)
        for line in (ROOT / "results/evaluation/evaluation_predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    pairs = {}
    for row in rows:
        key = row["label"], row["source_id"]
        if key in pairs:
            raise ValueError("Duplicate response")
        pairs[key] = row
    ids = [q["source_id"] for q in protocol["questions"]]
    if set(pairs) != {(arm, i) for arm in ("base", "tuned") for i in ids}:
        raise ValueError("Missing or unexpected paired response")
    base = [
        score(pairs["base", i]["response"], pairs["base", i]["gold"])["correct"]
        for i in ids
    ]
    tuned = [
        score(pairs["tuned", i]["response"], pairs["tuned", i]["gold"])["correct"]
        for i in ids
    ]
    report = paired_outcomes(base, tuned)
    changed = []
    for q, b, t in zip(protocol["questions"], base, tuned):
        i = q["source_id"]
        if pairs["base", i]["response"] != pairs["tuned", i]["response"]:
            changed.append(
                {
                    "source_id": i,
                    "question": q["question"],
                    "gold": q["gold"],
                    "base_correct": b,
                    "tuned_correct": t,
                    "base_response": pairs["base", i]["response"],
                    "tuned_response": pairs["tuned", i]["response"],
                }
            )
    report["response_texts_changed"] = len(changed)
    report["interpretation"] = (
        "Tagged-answer metric on 32 development questions; not evidence of improved mathematical reasoning."
    )
    return report, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for JSON report and changed-response review",
    )
    args = parser.parse_args()
    report, changed = analyze()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "paired_analysis.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        (args.output / "changed_responses.json").write_text(
            json.dumps(changed, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
