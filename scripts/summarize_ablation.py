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


def load_rows(run):
    path = run / "training_responses.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def matched_steps(historical_run, corrected_run):
    """Steps where both arms sampled identical responses, so only the reward differs.

    Same seed means identical generations until the adapters diverge. On those
    steps, any reward difference comes from the parser alone.
    """
    groups = {}
    for old, new in zip(load_rows(historical_run), load_rows(corrected_run)):
        if old["source_id"] != new["source_id"] or old["call_id"] != new["call_id"]:
            raise ValueError("Arms saw different questions; not comparable")
        groups.setdefault(old["call_id"], []).append((old, new))
    steps = []
    for call_id, pairs in sorted(groups.items()):
        if not all(o["response"] == n["response"] for o, n in pairs):
            break
        steps.append(
            dict(
                step=call_id,
                source_id=pairs[0][0]["source_id"],
                historical_rewards=[o["total_reward"] for o, _ in pairs],
                corrected_rewards=[n["total_reward"] for _, n in pairs],
                penalized_correct=[
                    o["response"][-40:]
                    for o, n in pairs
                    if n["corrected_correct"] and not o["historical_correct"]
                ],
            )
        )
    return steps


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
    report = dict(runs=results)
    old, new = (a.runs / f"{arm}_seed{a.seed}" for arm in ("historical", "corrected"))
    if (old / "diagnostics_summary.json").exists() and (
        new / "diagnostics_summary.json"
    ).exists():
        steps = matched_steps(old, new)
        report["matched_steps"] = steps
        differing = [s for s in steps if s["historical_rewards"] != s["corrected_rewards"]]
        print(
            f"\nIdentical samples in historical and corrected arms for the first "
            f"{len(steps)} steps; rewards differed on {len(differing)} of them:"
        )
        for s in differing:
            print(
                f"  step {s['step']}: historical {s['historical_rewards']} vs "
                f"corrected {s['corrected_rewards']}; old parser rejected "
                f"correct answers ending {s['penalized_correct']}"
            )
    if a.output:
        if a.output.exists():
            sys.exit(f"Refusing to overwrite {a.output}")
        a.output.write_text(json.dumps(report, indent=2))
    if len(results) < len(ARMS) or not all(r["feasible"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
