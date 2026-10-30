import unittest
from paired_analysis import paired_outcomes


class PairedAnalysisTests(unittest.TestCase):
    def test_two_gains_exact_test(self):
        r = paired_outcomes([False, False], [True, True], bootstrap_samples=100)
        self.assertEqual(r["exact_mcnemar_two_sided_p"], 0.5)
        self.assertEqual(r["difference_percentage_points"], 100)

    def test_no_discordance(self):
        r = paired_outcomes([True, False], [True, False], bootstrap_samples=100)
        self.assertEqual(r["exact_mcnemar_two_sided_p"], 1)
        self.assertEqual(r["paired_bootstrap_95_percentile_pp"], [0, 0])

    def test_reversal(self):
        b, t = [True, False, False], [False, True, True]
        forward = paired_outcomes(b, t, bootstrap_samples=100)
        reverse = paired_outcomes(t, b, bootstrap_samples=100)
        self.assertEqual(forward["gains"], reverse["losses"])
        self.assertEqual(
            forward["difference_percentage_points"],
            -reverse["difference_percentage_points"],
        )
        self.assertEqual(
            forward["exact_mcnemar_two_sided_p"], reverse["exact_mcnemar_two_sided_p"]
        )

    def test_invalid_inputs(self):
        for b, t in [([], []), ([True], []), ([1], [True])]:
            with self.assertRaises(ValueError):
                paired_outcomes(b, t)
