import unittest
from rewards import *


def wrap(a="42", r="First line\nSecond line"):
    return f"<reasoning>{r}</reasoning>\n<answer>\n{a}\n</answer>"


class RewardsTest(unittest.TestCase):
    def test_multiline(self):
        self.assertTrue(score(wrap(), "42")["structured_correct"])

    def test_plain(self):
        self.assertFalse(score("42", "42")["correct"])

    def test_missing_close(self):
        self.assertIsNone(extract_answer("<answer>42"))

    def test_missing_open(self):
        self.assertIsNone(extract_answer("42</answer>"))

    def test_duplicate(self):
        self.assertFalse(score(wrap() + "<answer>42</answer>", "42")["correct"])

    def test_trailing(self):
        self.assertFalse(score(wrap() + " actually 43", "42")["correct"])

    def test_empty(self):
        self.assertIsNone(parse(wrap(r=" ")))

    def test_wrong(self):
        self.assertFalse(score(wrap("41"), "42")["correct"])

    def test_equivalent(self):
        self.assertTrue(score(wrap("1,200.0"), "1200")["correct"])

    def test_negative(self):
        self.assertTrue(score(wrap("-42"), "-42")["correct"])

    def test_malformed_number(self):
        self.assertIsNone(number("12,00"))

    def test_nan(self):
        self.assertIsNone(number("NaN"))

    def test_units(self):
        self.assertIsNone(number("42 dollars"))

    def test_gold(self):
        self.assertEqual(gold_answer("work\n#### 42"), "42")

    def test_batch(self):
        self.assertEqual(correctness_reward([[{"content": wrap()}]], ["42"]), [2.0])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            correctness_reward([wrap()], [])

    def test_independent(self):
        s = score("Some work\n<answer>87</answer>", "87")
        self.assertTrue(s["correct"])
        self.assertFalse(s["format_valid"])

    def test_currency(self):
        self.assertTrue(score("<answer>$15</answer>", "15")["correct"])

    def test_markdown(self):
        self.assertTrue(score("**<answer>58</answer>**", "58")["correct"])

    def test_multiple_numbers(self):
        self.assertFalse(score("<answer>42 or 43</answer>", "42")["correct"])

    def test_nested(self):
        self.assertFalse(score("<answer><answer>42</answer></answer>", "42")["correct"])

    def test_intermediate(self):
        self.assertFalse(
            score("42 is intermediate; final result unknown", "42")["correct"]
        )

    def test_mismatched_order(self):
        self.assertIsNone(extract_answer("</answer>42<answer>"))

    def test_incomplete_no_reward(self):
        self.assertEqual(correctness_reward(["<answer>4"], ["4"]), [0.0])


if __name__ == "__main__":
    unittest.main()
