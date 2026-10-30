import json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from diagnostics import make_logged_reward, summarize
from rewards import correctness_reward


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name) / "log.jsonl"

    def test_preserves_rewards_and_logs(self):
        f = make_logged_reward(self.p)
        texts = ["<answer>42</answer>", "<answer>41</answer>"]
        result = f(
            prompts=["q", "q"],
            completions=texts,
            answer=["42", "42"],
            source_id=[9, 9],
            completion_ids=[[1], [2]],
            trainer_state=SimpleNamespace(global_step=0),
        )
        self.assertEqual(result, correctness_reward(texts, ["42", "42"]))
        rows = [json.loads(l) for l in self.p.read_text().splitlines()]
        self.assertEqual(rows[0]["source_id"], 9)
        self.assertEqual(rows[0]["optimizer_steps_completed"], 0)
        summary = summarize(self.p)
        self.assertEqual(summary["groups_with_reward_variation"], 1)

    def test_no_overwrite(self):
        self.p.write_text("original")
        with self.assertRaises(FileExistsError):
            make_logged_reward(self.p)
        self.assertEqual(self.p.read_text(), "original")

    def test_failed_parse_is_not_math_failure(self):
        f = make_logged_reward(self.p)
        f(["q"], ["Final answer: 42"], ["42"])
        self.assertEqual(
            summarize(self.p)["status_counts"], {"missing_or_ambiguous_answer_field": 1}
        )

    def test_mismatch_fails(self):
        f = make_logged_reward(self.p)
        with self.assertRaises(ValueError):
            f([], ["x"], ["42"])

    def test_multiple_calls(self):
        f = make_logged_reward(self.p)
        for _ in range(2):
            f(["q"], ["<answer>1</answer>"], ["1"])
        self.assertEqual(summarize(self.p)["reward_calls"], 2)

    def test_format_reward_and_tokens(self):
        f = make_logged_reward(self.p, max_tokens=2)
        f(
            ["q"],
            ["<reasoning>x</reasoning><answer>1</answer>"],
            ["1"],
            completion_ids=[[1, 2]],
        )
        row = json.loads(self.p.read_text())
        self.assertEqual(row["total_reward"], 2.5)
        self.assertTrue(row["hit_token_cap"])


if __name__ == "__main__":
    unittest.main()
