"""Compare reward-ablation runs on training-side evidence. CPU only.

Reads run directories named <arm>_seed<seed> produced by scripts/train_arm.py.
These are sampled training responses, not held-out accuracy.
"""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_rewards import ARMS


def summarize_run(run, num_generations):
    rows = [
        json.loads(line)
        for line in (run / "training_responses.jsonl").read_text().splitlines()
        if line.strip()
    ]
    diag = json.loads((run / "diagnostics_summary.json").read_text())
    training = json.loads((run / "training.json").read_text())
    n = len(rows)
    tokens = [r["generated_tokens"] for r in rows if r["generated_tokens"] is not None]
    out = dict(
        arm=diag["arm"],
        seed=diag["seed"],
        completed_steps=diag["completed_steps"],
        requested_steps=diag["requested_steps"],
        seconds=round(training["seconds"], 1),
        peak_allocated_gib=round(training["peak_allocated_gib"], 2),
        responses=n,
        unique_questions=len({r["source_id"] for r in rows}),
        historical_correct=sum(r["historical_correct"] for r in rows),
        corrected_correct=sum(r["corrected_correct"] for r in rows),
        parser_disagreements=sum(
            r["historical_correct"] != r["corrected_correct"] for r in rows
        ),
        format_valid=sum(r["format_valid"] for r in rows),
        hit_token_cap=sum(r["hit_token_cap"] is True for r in rows),
        mean_tokens=round(sum(tokens) / len(tokens), 1) if tokens else None,
        mean_reward=round(sum(r["total_reward"] for r in rows) / n, 4) if n else None,
        groups_with_reward_variation=diag["groups_with_reward_variation"],
        prompt_groups=diag["prompt_groups"],
        steps_with_nonzero_gradient=diag["steps_with_nonzero_gradient"],
        tensors_changed=diag["tensors_changed"],
        tensors_checked=diag["tensors_checked"],
    )
    checks = {
        "completed_all_steps": diag["complete"],
        "every_step_logged": n == diag["completed_steps"] * num_generations,
        "nonzero_gradient_seen": diag["steps_with_nonzero_gradient"] > 0,
        "all_adapter_tensors_changed": diag["tensors_changed"] == diag["tensors_checked"],
        "arm_matches_logs": all(r["arm"] == diag["arm"] for r in rows),
    }
    out["checks"] = checks
    out["feasible"] = all(checks.values())
    return out


COLUMNS = [
    ("arm", "Arm"),
    ("completed_steps", "Steps"),
    ("seconds", "Seconds"),
    ("peak_allocated_gib", "Peak GiB"),
    ("corrected_correct", "Correct (corrected)"),
    ("historical_correct", "Correct (historical)"),
    ("format_valid", "Strict format"),
    ("groups_with_reward_variation", "Groups w/ variation"),
    ("steps_with_nonzero_gradient", "Nonzero-grad steps"),
    ("hit_token_cap", "Token cap"),
    ("feasible", "Feasible"),
]


def table(results):
    lines = ["| " + " | ".join(h for _, h in COLUMNS) + " |"]
    lines.append("|" + "---|" * len(COLUMNS))
    for r in results:
        cells = []
        for key, _ in COLUMNS:
            value = r[key]
            if key in ("corrected_correct", "historical_correct", "format_valid"):
                value = f"{value}/{r['responses']}"
            if key == "groups_with_reward_variation":
                value = f"{value}/{r['prompt_groups']}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Directory holding <arm>_seed<seed>/")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    a = parser.parse_args()
    plan = json.loads((ROOT / "experiments/reward_ablation.json").read_text())
    results = []
    for arm in ARMS:
        run = a.runs / f"{arm}_seed{a.seed}"
        if not (run / "diagnostics_summary.json").exists():
            print(f"MISSING: {run} has not finished")
            continue
        results.append(summarize_run(run, plan["num_generations"]))
    if not results:
        sys.exit("No finished runs found")
    print(table(results))
    print("\nTraining samples only; not held-out accuracy or evidence of reasoning.")
    if a.output:
        if a.output.exists():
            sys.exit(f"Refusing to overwrite {a.output}")
        a.output.write_text(json.dumps(results, indent=2))
    if len(results) < len(ARMS) or not all(r["feasible"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
