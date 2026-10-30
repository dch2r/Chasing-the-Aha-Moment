import json
import tempfile
import unittest
from pathlib import Path

from arm_logging import make_arm_logged_reward
from diagnostics import summarize

PERIOD = "<reasoning>2 + 2 = 4.</reasoning><answer>4</answer>."
CLEAN = "<reasoning>2 + 2 = 4.</reasoning><answer>4</answer>"


class ArmLoggingTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "log.jsonl"

    def tearDown(self):
        self.dir.cleanup()

    def rows(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_returns_arm_correctness_only(self):
        for arm, expected in [("historical", 0), ("corrected", 2), ("format_only", 0)]:
            path = Path(self.dir.name) / f"{arm}.jsonl"
            reward = make_arm_logged_reward(path, arm)
            self.assertEqual(reward(["q"], [PERIOD], ["4"]), [expected])

    def test_logs_both_parsers_for_every_arm(self):
        reward = make_arm_logged_reward(self.path, "format_only")
        reward(["q", "q"], [PERIOD, CLEAN], ["4", "4"], source_id=[7, 7])
        first, second = self.rows()
        self.assertFalse(first["historical_correct"])
        self.assertTrue(first["corrected_correct"])
        self.assertTrue(second["historical_correct"])
        self.assertEqual(first["arm"], "format_only")
        self.assertEqual(first["correctness_reward"], 0)
        self.assertEqual(second["total_reward"], 0.5)
        self.assertEqual(first["source_id"], 7)

    def test_compatible_with_pilot_summary(self):
        reward = make_arm_logged_reward(self.path, "corrected", max_tokens=2)
        reward(["q", "q"], [PERIOD, "no tags"], ["4", "4"], completion_ids=[[1, 2], [1]])
        summary = summarize(self.path)
        self.assertEqual(summary["logged_completions"], 2)
        self.assertEqual(summary["groups_with_reward_variation"], 1)
        self.assertEqual(summary["hit_token_cap"], 1)
        self.assertEqual(
            summary["status_counts"],
            {"correct": 1, "missing_or_duplicate_answer_tags": 1},
        )

    def test_refuses_existing_log_and_bad_input(self):
        self.path.write_text("original")
        with self.assertRaises(FileExistsError):
            make_arm_logged_reward(self.path, "corrected")
        self.assertEqual(self.path.read_text(), "original")
        with self.assertRaises(ValueError):
            make_arm_logged_reward(Path(self.dir.name) / "x.jsonl", "unknown")
        reward = make_arm_logged_reward(Path(self.dir.name) / "y.jsonl", "corrected")
        with self.assertRaises(ValueError):
            reward([], [CLEAN], ["4"])


if __name__ == "__main__":
    unittest.main()
