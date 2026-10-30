import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExperimentPlanTests(unittest.TestCase):
    def test_proposed_split_excludes_all_seen_examples(self):
        plan = json.loads((ROOT / "experiments/reward_ablation.json").read_text())
        old = json.loads((ROOT / "results/evaluation/protocol.json").read_text())
        manifest = json.loads(
            (ROOT / "results/pilot/run_metadata/data_manifest.json").read_text()
        )
        self.assertEqual(plan["train_ids"], manifest["train_ids"])
        excluded = set(
            manifest["train_ids"]
            + manifest["dev_ids"]
            + [q["source_id"] for q in old["questions"]]
        )
        selected = plan["evaluation_ids"]
        self.assertEqual(len(selected), 256)
        self.assertEqual(len(set(selected)), 256)
        self.assertTrue(set(selected).isdisjoint(excluded))
        self.assertTrue(all(0 <= i < 7473 for i in selected))
        self.assertEqual(plan["revision"], old["revision"])
        self.assertEqual(plan["status"], "stage1_complete_stage2_not_run")
