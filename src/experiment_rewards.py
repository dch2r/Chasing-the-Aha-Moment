"""Prospective reward arms. Never used to relabel the historical training run."""

from eval_scoring import score as corrected_score
from rewards import contents, score as historical_score

ARMS = ("historical", "corrected", "format_only")


def reward_components(text, gold, arm):
    """Change correctness parsing only between historical and corrected arms."""
    if arm not in ARMS:
        raise ValueError(f"Unknown reward arm: {arm}")
    # Keep format reward identical across all arms to isolate correctness parsing.
    old = historical_score(text, gold)
    correctness = 0.0
    if arm == "historical":
        correctness = 2.0 * old["correct"]
    elif arm == "corrected":
        correctness = 2.0 * corrected_score(text, gold)["correct"]
    formatting = 0.5 * old["format_valid"]
    return {
        "correctness_reward": correctness,
        "format_reward": formatting,
        "total_reward": correctness + formatting,
    }


def make_reward(arm):
    """TRL-compatible reward; the experiment runner must separately log samples."""
    if arm not in ARMS:
        raise ValueError(f"Unknown reward arm: {arm}")

    def reward(completions, answer, **kwargs):
        texts = contents(completions)
        if len(texts) != len(answer):
            raise ValueError("Completion/answer lengths differ")
        return [
            reward_components(text, gold, arm)["total_reward"]
            for text, gold in zip(texts, answer)
        ]

    reward.__name__ = f"{arm}_reward"
    return reward
