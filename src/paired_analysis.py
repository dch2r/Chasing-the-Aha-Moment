"""Paired binary-outcome analysis without third-party dependencies."""

import math
import random


def paired_outcomes(base, tuned, *, bootstrap_samples=10000, seed=3407):
    """Return paired counts, a percentile bootstrap interval and exact McNemar p.

    The exact test conditions on discordant pairs. The bootstrap resamples
    question pairs, never the two model arms independently. These statistics
    describe the chosen binary metric; they cannot establish reasoning gains.
    """
    if not base or len(base) != len(tuned):
        raise ValueError("Require equal, nonempty paired outcomes")
    if any(type(x) is not bool for x in [*base, *tuned]):
        raise ValueError("Outcomes must be booleans")
    if bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    differences = [int(t) - int(b) for b, t in zip(base, tuned)]
    n = len(differences)
    wins, losses = differences.count(1), differences.count(-1)
    discordant = wins + losses
    p_value = min(
        1.0,
        2
        * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1))
        / 2**discordant,
    )
    rng = random.Random(seed)
    boot = sorted(
        100 * sum(rng.choices(differences, k=n)) / n for _ in range(bootstrap_samples)
    )
    return {
        "n": n,
        "base_correct": sum(base),
        "tuned_correct": sum(tuned),
        "both_correct": sum(b and t for b, t in zip(base, tuned)),
        "both_incorrect": sum(not b and not t for b, t in zip(base, tuned)),
        "gains": wins,
        "losses": losses,
        "difference_percentage_points": 100 * sum(differences) / n,
        "paired_bootstrap_95_percentile_pp": [
            boot[int(0.025 * (bootstrap_samples - 1))],
            boot[int(0.975 * (bootstrap_samples - 1))],
        ],
        "exact_mcnemar_two_sided_p": p_value,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }
