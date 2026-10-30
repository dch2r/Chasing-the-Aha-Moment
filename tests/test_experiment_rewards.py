import unittest
from experiment_rewards import reward_components, make_reward


class ExperimentRewardsTests(unittest.TestCase):
    def test_corrected_punctuation(self):
        text = "<reasoning>2 + 2 = 4.</reasoning><answer>4</answer>."
        old = reward_components(text, "4", "historical")
        new = reward_components(text, "4", "corrected")
        self.assertEqual(old["correctness_reward"], 0)
        self.assertEqual(new["correctness_reward"], 2)
        self.assertEqual(old["format_reward"], new["format_reward"])

    def test_format_control_ignores_correctness(self):
        text = "<reasoning>Calculation.</reasoning><answer>99</answer>"
        self.assertEqual(
            reward_components(text, "4", "format_only")["total_reward"], 0.5
        )

    def test_wrong_answer_not_rewarded(self):
        for arm in ["historical", "corrected"]:
            self.assertEqual(
                reward_components("<answer>99</answer>", "4", arm)[
                    "correctness_reward"
                ],
                0,
            )

    def test_ambiguous_answer_rejected(self):
        text = "<answer>4</answer><answer>5</answer>"
        self.assertEqual(
            reward_components(text, "4", "corrected")["correctness_reward"], 0
        )

    def test_wrapper_and_errors(self):
        self.assertEqual(make_reward("corrected")(["<answer>4</answer>."], ["4"]), [2])
        with self.assertRaises(ValueError):
            make_reward("unknown")
        with self.assertRaises(ValueError):
            make_reward("corrected")([], ["4"])


class RecordedRewardParityTests(unittest.TestCase):
    def test_historical_arm_matches_recorded_rewards(self):
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "results/pilot/training_responses.jsonl"
        )
        for line in path.read_text().splitlines():
            row = json.loads(line)
            actual = reward_components(row["response"], row["gold"], "historical")
            self.assertEqual(actual["total_reward"], row["total_reward"])
            self.assertEqual(actual["format_reward"], row["format_reward"])
