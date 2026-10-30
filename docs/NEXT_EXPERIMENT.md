# Next experiment: does reward parsing change learning?

Status: protocol, reward components and the stage 1 runner are prepared and CPU-tested; no ablation training or new evaluation has been run yet. This is the substantive next step if the goal is a stronger research project. README polishing alone cannot supply new evidence.

## Hypothesis and controls

A parser that rejects correct numeric answer fields because of punctuation can change GRPO advantages. Compare three fresh adapters initialized from the same pinned Qwen base model:

1. `historical`: historical numeric correctness reward plus the historical format reward.
2. `corrected`: corrected numeric correctness reward plus the exact same format reward.
3. `format_only`: the same format reward, with correctness reward zero.

The untrained base model is an additional evaluation arm. The format-only control tests whether any gain can be explained by formatting alone. These adapters must be trained from the base, not continued from the current saved adapter. The old adapter is historical evidence, not a substitute for a matched experimental run.

The components in `src/experiment_rewards.py` are tested. The historical arm is checked against all 80 recorded rewards. `scripts/train_arm.py` trains one arm and seed with the pilot's settings, and `src/arm_logging.py` scores every sampled response with both parsers regardless of arm. `notebooks/Reward_Ablation_Stage1.ipynb` runs stage 1 on a Colab T4, and `scripts/summarize_ablation.py` compares the arms. The runner has not yet been run on a GPU.

## Fixed design

`experiments/reward_ablation.json` records the base revision, training IDs, new development IDs, arm definitions, seeds and budgets. The 256 proposed evaluation IDs are selected deterministically from GSM8K's TRAIN split, excluding all 128 training, eight earlier development and 32 already-inspected evaluation examples. The new question contents have not been examined here. Verify the dataset fingerprint before loading their contents.

Stage 1: a 20-step feasibility run for each arm, seed 3407, without inspecting the reserved 256-question set. Check gradients, reward logs, memory and all expected files. These feasibility adapters are discarded before the main comparison.

Stage 2: 100 steps per arm for seeds 3407, 3408 and 3409: nine fresh training runs. Hold model revision, data order per seed, LoRA targets/rank, optimizer, learning rate, four generations and generation limits fixed. A time cutoff is an incomplete run, not permission to compare unequal training budgets. Record wall time and generated token counts; report their differences.

The present 20-step run took about 13 minutes and its 64-answer replay took over 20 minutes. Nine longer runs plus evaluation represent a substantial GPU commitment; measure stage 1 before scheduling stage 2. No precise runtime promise is justified.

## Evaluation and decision rule

Freeze the evaluation prompt, greedy decoding and parser before main training. Use the same 256 questions across base and every adapter. Report tagged accuracy, parseable coverage, full-format compliance, truncation, response length and paired differences per seed. Do not pool three seed outcomes as 768 independent questions. Report seed variability and question-paired uncertainty explicitly.

Before looking at gold labels or model identity, review final numeric answers in outputs that the parser rejects. Use a fixed written adjudication rule and document ambiguous cases. This complementary, format-independent review is needed to discuss mathematical answers; a higher tagged score alone cannot support a reasoning claim. If a second reviewer is unavailable, report that limitation rather than imply independent adjudication.

Success is a reproducible explanation of how rewards affect outputs, even if gains are null. A stronger reasoning claim would require consistent improvements beyond the format-only control under the format-independent assessment. Keep the official GSM8K TEST split untouched until development choices are finalized.

## Llama extension

Only after the controlled Qwen study, consider an 8B replication to test whether findings transfer. Do not rerun the historical notebook unchanged. If an old Llama adapter is available, a matched evaluation of that artifact is a cheaper first step. See `LLAMA_REVIEW.md`.
