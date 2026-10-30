"""Verify recorded evidence without model downloads or GPU access."""

import ast, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eval_scoring import score
from rewards import score as training_score
from diagnostics import summarize

p = ROOT / "results/evaluation"
protocol_text = (p / "protocol.json").read_text()
protocol = json.loads(protocol_text)
summary = json.loads((p / "evaluation_summary.json").read_text())
hash_ = hashlib.sha256(protocol_text.encode()).hexdigest()
assert hash_ == summary["protocol_hash"]
assert (
    hashlib.sha256((ROOT / "src/eval_scoring.py").read_bytes()).hexdigest()
    == protocol["scorer_sha256"]
)
questions = {x["source_id"]: x for x in protocol["questions"]}
rows = [
    json.loads(l) for l in (p / "evaluation_predictions.jsonl").read_text().splitlines()
]
assert len(rows) == 64 and len(questions) == 32 and summary["complete"]
paired = {}
for row in rows:
    key = (row["label"], row["source_id"])
    assert key not in paired
    assert (
        row["protocol_hash"] == hash_
        and row["gold"] == questions[row["source_id"]]["gold"]
    )
    scored = score(row["response"], row["gold"])
    for field, value in scored.items():
        assert row[field] == value, (key, field)
    paired[key] = row
for label in ["base", "tuned"]:
    selected = [paired[(label, i)] for i in questions]
    assert sum(r["correct"] for r in selected) == summary[label]["correct"]
    assert sum(r["answer_parseable"] for r in selected) == summary[label]["parseable"]
    assert sum(r["format_valid"] for r in selected) == summary[label]["format_valid"]
    assert sum(r["hit_token_cap"] for r in selected) == summary[label]["hit_token_cap"]
wins = sum(
    not paired["base", i]["correct"] and paired["tuned", i]["correct"]
    for i in questions
)
losses = sum(
    paired["base", i]["correct"] and not paired["tuned", i]["correct"]
    for i in questions
)
assert (wins, losses) == (summary["changed_to_correct"], summary["changed_to_wrong"])
t = ROOT / "results/pilot"
train = [
    json.loads(l) for l in (t / "training_responses.jsonl").read_text().splitlines()
]
assert {r["source_id"] for r in train}.isdisjoint(questions)
assert len(train) == 80
for r in train:
    s = training_score(r["response"], r["gold"])
    assert (
        r["correct"] == s["correct"]
        and r["total_reward"] == 2 * s["correct"] + 0.5 * s["format_valid"]
    )
diagnostic = json.loads((t / "diagnostics_summary.json").read_text())
for k, v in summarize(t / "training_responses.jsonl").items():
    assert diagnostic[k] == v, k
# Verify the recovered original training selection and configuration.
m = t / "run_metadata"
manifest = json.loads((m / "data_manifest.json").read_text())
config = json.loads((m / "config.json").read_text())
trainer_config = json.loads((m / "trainer_config.json").read_text())
revision = json.loads((m / "model_revision.json").read_text())
selected_train = [json.loads(l) for l in (m / "train.jsonl").read_text().splitlines()]
selected_dev = [json.loads(l) for l in (m / "dev.jsonl").read_text().splitlines()]
train_ids = [x["source_id"] for x in selected_train]
dev_ids = [x["source_id"] for x in selected_dev]
assert train_ids == manifest["train_ids"] and dev_ids == manifest["dev_ids"]
assert len(set(train_ids)) == len(train_ids) == config["train_size"] == 128
assert len(set(dev_ids)) == len(dev_ids) == config["dev_size"] == 8
assert set(train_ids).isdisjoint(dev_ids)
assert set(train_ids).isdisjoint(questions) and set(dev_ids).isdisjoint(questions)
assert {x["source_id"] for x in train} <= set(train_ids)
assert manifest["fingerprint"] == protocol["dataset_fingerprint"]
assert manifest["dataset"] == "openai/gsm8k" and manifest["split"] == "train"
assert revision["model"] == config["model"] == protocol["model"]
assert revision["resolved_commit"] == protocol["revision"]
assert config["steps"] == trainer_config["max_steps"] == 20
assert config["num_generations"] == trainer_config["num_generations"] == 4
assert config["seed"] == trainer_config["seed"] == protocol["seed"]
assert config["max_completion_length"] == trainer_config["max_completion_length"] == 512
for name in ["rewards.py", "diagnostics.py"]:
    assert (m / name).read_bytes() == (ROOT / "src" / name).read_bytes(), name
# Adapter is distributed in the ZIP but intentionally excluded from ordinary Git commits.
if (ROOT / "adapter").is_dir():
    from artifact_checks import verify_adapter

    verify_adapter(ROOT / "adapter", protocol)
    print("PASS: bundled adapter/tokenizer fingerprints match the recorded evaluation.")
print(
    "PASS: all 128 training and 8 prior development IDs are disjoint from the 32 evaluation IDs; original configuration and utility sources match."
)

# The standalone modules must match their notebook-embedded copies.
embedded = set()
for path in (ROOT / "notebooks").rglob("*.ipynb"):
    nb = json.loads(path.read_text())
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        source = "".join(c["source"])
        if source.startswith("%"):
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "write_text"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                obj = node.func.value
                if (
                    isinstance(obj, ast.Call)
                    and obj.args
                    and isinstance(obj.args[0], ast.Constant)
                ):
                    name = Path(obj.args[0].value).name
                    if (ROOT / "src" / name).exists():
                        assert (ROOT / "src" / name).read_text() == node.args[
                            0
                        ].value, name
                        embedded.add(name)
assert embedded == {
    "rewards.py",
    "diagnostics.py",
    "eval_scoring.py",
    "evaluation_io.py",
}
print(
    "PASS: 64 evaluation rows, protocol and scorer hashes, 80 training rewards, disjoint observed IDs, and embedded module consistency."
)
print(
    "Recorded tagged-answer scores: base 21/32; adapter 23/32. Not a reasoning improvement claim."
)

# Verify the completed portable T4 replay independently of its reported summary.
rp = ROOT / "results/replay"
replay_spec = json.loads((rp / "replay_protocol.json").read_text())
replay_hash = hashlib.sha256((rp / "replay_protocol.json").read_bytes()).hexdigest()
assert replay_spec["recorded_protocol_sha256"] == hash_
assert replay_spec["adapter_sha256"] == protocol["adapter_sha256"]
assert replay_spec["environment"] == protocol["versions"]
assert replay_spec["gpu"] == "Tesla T4"
replay_summary = json.loads((rp / "evaluation_summary.json").read_text())
assert replay_summary["complete"] and replay_summary["saved_answers"] == 64
assert (
    replay_summary["expected_questions"] == 32
    and replay_summary["protocol_hash"] == replay_hash
)
replay_rows = [
    json.loads(l)
    for l in (rp / "evaluation_predictions.jsonl").read_text().splitlines()
]
replayed = {}
for row in replay_rows:
    key = (row["label"], row["source_id"])
    assert key in paired and key not in replayed
    assert (
        row["gold"] == questions[row["source_id"]]["gold"]
        and row["protocol_hash"] == replay_hash
    )
    for field, value in score(row["response"], row["gold"]).items():
        assert row[field] == value, (key, field)
    assert row["hit_token_cap"] == (
        row["generated_tokens"] >= protocol["max_new_tokens"]
    )
    replayed[key] = row
assert len(replayed) == 64
for label in ["base", "tuned"]:
    selected = [replayed[label, i] for i in questions]
    for field in ["correct", "answer_parseable", "format_valid", "hit_token_cap"]:
        assert sum(x[field] for x in selected) == replay_summary[label][field]
    assert replay_summary[label]["accuracy"] == sum(x["correct"] for x in selected) / 32
changes = [
    int(replayed["tuned", i]["correct"]) - int(replayed["base", i]["correct"])
    for i in questions
]
assert changes.count(1) == replay_summary["changed_to_correct"] == 2
assert changes.count(-1) == replay_summary["changed_to_wrong"] == 0
assert (
    100 * sum(changes) / 32 == replay_summary["accuracy_difference_percentage_points"]
)
identical = sum(replayed[k]["response"] == paired[k]["response"] for k in paired)
print(
    f"PASS: complete T4 replay, all 64 scores checked; {identical}/64 response texts identical to original evaluation."
)

# Recompute the published CPU analysis rather than trusting its stored summary.
from analyze_results import analyze

analysis, changed = analyze()
assert analysis == json.loads(
    (ROOT / "results/analysis/paired_analysis.json").read_text()
)
assert changed == json.loads(
    (ROOT / "results/analysis/changed_responses.json").read_text()
)
print("PASS: paired analysis and changed-response export match recomputation.")
