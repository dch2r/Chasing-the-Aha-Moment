# Reward ablation, stage 1

Three fresh LoRA adapters were trained from the same Qwen2.5-1.5B-Instruct revision for 20 GRPO steps each on a Colab Tesla T4, seed 3407, using the pilot's 128 training questions and settings. Only the correctness reward differed:

- `historical`: the pilot's original parser, which rejects a correct tagged answer followed by a period or a unit.
- `corrected`: the parser used for evaluation.
- `format_only`: no correctness reward, as a control.

All three runs used repository commit `254c33a`. The logs, configurations and per-tensor adapter changes are in `results/ablation/stage1/`. The adapters were discarded as planned. Reproduce the summary with:

```bash
python scripts/summarize_ablation.py results/ablation/stage1
```

## Feasibility

| Arm | Seconds | Peak GiB | Steps with reward variation | Nonzero-gradient steps | Adapter tensors changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical | 772 | 3.20 | 17/20 | 17 | 224/224 |
| corrected | 799 | 3.17 | 15/20 | 15 | 224/224 |
| format_only | 757 | 3.21 | 7/20 | 7 | 224/224 |

Every arm completed 20 steps with every sampled response logged and no truncated outputs.

## What the runs show

**The old parser flips the training signal on correct answers.** With the same seed, the historical and corrected arms sampled word-for-word identical responses for the first five steps, so on those steps the reward function was the only difference. On two of the five, the historical parser gave zero reward to a correct answer that the corrected parser credited:

- Step 1: `…the answer is <answer>407</answer>.` Rewards `[2, 0, 2, 2.5]` versus `[2, 2, 2, 2.5]`.
- Step 4: `<answer>25 hours</answer>`. Rewards `[0, 0, 2, 0]` versus `[0, 0, 2, 2]`.

GRPO scores each response relative to its group, so in step 1 the historical reward pushed the model away from a correct answer only because of its trailing period. Over the full historical run, 7 of 80 sampled responses were correct but received no correctness reward.

**Format reward alone is a sparse signal.** The format-only control produced differing rewards within a group, and therefore a nonzero gradient, on only 7 of 20 steps, against 15 to 17 for the arms with a correctness reward.

## What it does not show

Correct-answer counts on training samples (42, 30 and 35 out of 80 by the corrected parser) are not comparable between arms. After step 5 the arms sample different responses, and 80 samples from 20 questions are too few to compare. No held-out evaluation was run, so stage 1 makes no claim about accuracy or reasoning.
