# Interpretation of the recorded results

## Two score changes

- Source ID 6762, gold 10: both outputs calculate ten cats. The base ends with an untagged answer; the adapter provides `<answer>10</answer>`.
- Source ID 7315, gold 44: both outputs calculate 44 roses. The adapter adds `<answer>44</answer>`.

The +6.25 percentage-point tagged-answer score difference therefore does not demonstrate better math. Full-format compliance was 1/32 versus 2/32; parseable answers were 23/32 versus 25/32. Only three response texts changed.

## What the pilot established

The GPU path completed, sampling and reward logs were captured, 14 optimizer steps had nonzero gradients, and all 224 LoRA tensors changed. The adapter was subsequently evaluated using matched decoding and questions excluded from training and the prior eight-question checks.

## What remains unresolved

Only 32 development questions were evaluated, with one seed and a short training run. The adapter trained against a scorer with known presentation-related false negatives. Improvements in tagged output compliance do not isolate reasoning ability. Comparing the four-response pilot with earlier two-response diagnostics is not a controlled ablation: sample sizes, schedules and random-number consumption differ.

## Defensible next experiment

Freeze a corrected training reward and evaluation policy before new runs. Compare base versus trained adapters with fixed compute budgets, disjoint development data, and preferably multiple seeds. Report parseable coverage and manual adjudication separately. Keep the official test split untouched until development choices are frozen. Do not relabel ambiguous questions to match generated answers.

