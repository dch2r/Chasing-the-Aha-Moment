"""Per-response logging for reward-ablation runs.

Every sampled response is scored by both the historical and corrected parsers,
whichever arm is training, so the arms can be compared on the same footing.
"""

import json, os
from pathlib import Path

from eval_scoring import score as corrected_score
from experiment_rewards import ARMS, reward_components
from rewards import contents, score as historical_score


def make_arm_logged_reward(path, arm, max_tokens=512):
    """Return a TRL correctness reward for `arm` that logs each sampled response.

    The format reward is passed to the trainer separately, as in the pilot, so the
    returned values are the arm's correctness component only.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown reward arm: {arm}")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Log already exists. Start a new run directory.")
    path.touch()
    call_id = 0

    def reward(
        prompts,
        completions,
        answer,
        completion_ids=None,
        source_id=None,
        trainer_state=None,
        **kwargs,
    ):
        nonlocal call_id
        texts = contents(completions)
        n = len(texts)
        if len(prompts) != n or len(answer) != n:
            raise ValueError("Prompt/completion/answer lengths differ")
        if source_id is not None and len(source_id) != n:
            raise ValueError("source_id length mismatch")
        if completion_ids is not None and len(completion_ids) != n:
            raise ValueError("completion_ids length mismatch")
        call_id += 1
        rows, out = [], []
        for i, (prompt, text, gold) in enumerate(zip(prompts, texts, answer)):
            parts = reward_components(text, gold, arm)
            old = historical_score(text, gold)
            new = corrected_score(text, gold)
            tokens = len(completion_ids[i]) if completion_ids is not None else None
            if new["correct"]:
                status = "correct"
            elif new["answer_parseable"]:
                status = "parsed_answer_mismatch"
            else:
                status = new["extraction_status"]
            rows.append(
                dict(
                    arm=arm,
                    call_id=call_id,
                    completion_index=i,
                    optimizer_steps_completed=getattr(
                        trainer_state, "global_step", None
                    ),
                    source_id=int(source_id[i]) if source_id is not None else None,
                    prompt=prompt,
                    gold=str(gold),
                    response=text,
                    status=status,
                    generated_tokens=tokens,
                    hit_token_cap=tokens >= max_tokens if tokens is not None else None,
                    historical_correct=old["correct"],
                    corrected_correct=new["correct"],
                    corrected_answer=new["extracted_answer"],
                    format_valid=old["format_valid"],
                    **parts,
                )
            )
            out.append(parts["correctness_reward"])
        with path.open("a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return out

    reward.__name__ = f"logged_{arm}_correctness"
    return reward
