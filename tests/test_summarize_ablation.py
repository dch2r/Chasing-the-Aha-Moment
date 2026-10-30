import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from arm_logging import make_arm_logged_reward
from diagnostics import summarize

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "summarize_ablation", ROOT / "scripts/summarize_ablation.py"
)
summarize_ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_ablation)

PERIOD = "<reasoning>2 + 2 = 4.</reasoning><answer>4</answer>."
CLEAN = "<reasoning>2 + 2 = 4.</reasoning><answer>4</answer>"


def fake_run(root, arm, steps=2, tensors_changed=4):
    run = root / f"{arm}_seed3407"
    run.mkdir()
    reward = make_arm_logged_reward(run / "training_responses.jsonl", arm)
    for step in range(steps):
        reward(["q"] * 4, [PERIOD, CLEAN, "no tags", CLEAN], ["4"] * 4,
               source_id=[step] * 4, completion_ids=[[1]] * 4)
    diag = summarize(run / "training_responses.jsonl")
    diag.update(arm=arm, seed=3407, requested_steps=steps, completed_steps=steps,
                complete=True, steps_with_nonzero_gradient=steps,
                tensors_checked=4, tensors_changed=tensors_changed)
    (run / "diagnostics_summary.json").write_text(json.dumps(diag))
    (run / "training.json").write_text(
        json.dumps(dict(seconds=1.0, peak_allocated_gib=3.0)))
    return run


class SummarizeAblationTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_counts_and_feasibility(self):
        result = summarize_ablation.summarize_run(fake_run(self.root, "corrected"), 4)
        self.assertEqual(result["responses"], 8)
        self.assertEqual(result["corrected_correct"], 6)
        self.assertEqual(result["historical_correct"], 4)
        self.assertEqual(result["parser_disagreements"], 2)
        self.assertEqual(result["unique_questions"], 2)
        self.assertTrue(result["feasible"])
        self.assertIn("6/8", summarize_ablation.table([result]))

    def test_unchanged_adapter_is_not_feasible(self):
        run = fake_run(self.root, "format_only", tensors_changed=3)
        result = summarize_ablation.summarize_run(run, 4)
        self.assertFalse(result["checks"]["all_adapter_tensors_changed"])
        self.assertFalse(result["feasible"])

    def test_missing_logged_step_is_not_feasible(self):
        result = summarize_ablation.summarize_run(fake_run(self.root, "historical"), 2)
        self.assertFalse(result["checks"]["every_step_logged"])


if __name__ == "__main__":
    unittest.main()
