import unittest
from eval_scoring import score


class PolicyTests(unittest.TestCase):
    def test_equivalent_presentations(self):
        cases = [
            ("<answer>25</answer>", "25"),
            ("<answer>25</answer>.", "25"),
            ("<answer>25 hours</answer>", "25"),
            ("<answer>25</answer> hours.", "25"),
            ("**<answer>25</answer>**", "25"),
            ("**<answer>25</answer>**.", "25"),
            ("<answer>$1,200.00</answer>", "1200"),
            ("<answer>-2.5</answer>", "-2.5"),
            ("<answer>20%</answer>", "20"),
            ("<answer>265 square feet</answer>", "265"),
            ("<answer>**25**</answer>", "25"),
            ("<answer>25.</answer>", "25"),
        ]
        for text, gold in cases:
            with self.subTest(text=text):
                self.assertTrue(score(text, gold)["correct"])

    def test_rejections(self):
        cases = [
            "25",
            "The answer is 25.",
            "<answer>25",
            "25</answer>",
            "<answer>25</answer><answer>25</answer>",
            "<answer>25</answer> or 26",
            "<answer>25 or 26</answer>",
            "<answer>2,5</answer>",
            "<answer>NaN</answer>",
            "<answer>2+23</answer>",
            "<answer>25/1</answer>",
            "<answer>2.5e1</answer>",
            "<answer>25 unknownunit</answer>",
            "<answer>25</answer> actually 26",
            "</answer>25<answer>",
            "<answer><answer>25</answer></answer>",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(score(text, "25")["answer_parseable"])

    def test_wrong_remains_wrong(self):
        self.assertFalse(score("<answer>26 hours</answer>.", "25")["correct"])

    def test_format_independent(self):
        s = score("Work\n<answer>25</answer>.", "25")
        self.assertTrue(s["correct"])
        self.assertFalse(s["format_valid"])

    def test_full_format(self):
        self.assertTrue(
            score("<reasoning>work\nwork</reasoning><answer>25</answer>", "25")[
                "structured_correct"
            ]
        )

    def test_never_gold_search(self):
        self.assertFalse(
            score("Intermediate value 25.\n<answer>26</answer>", "25")["correct"]
        )


if __name__ == "__main__":
    unittest.main()
