# When formatting looks like reasoning improvement

## Question

Can a short GRPO run on a modest GPU produce a measurable improvement, and can the evaluation distinguish mathematical correctness from formatting compliance?

The work began with an earlier Llama 3.1 8B GRPO experiment. Its benchmark claims could not be reconstructed from the available notebooks. A separate Qwen2.5 1.5B experiment was therefore built for limited Colab T4 access. Its results must not be attributed to the earlier Llama model.

## Implementation

The pilot uses four-bit NF4 quantization, rank-16 LoRA, and TRL GRPO. It logs every sampled response, the reference answer, parser output, reward components, gradient norms and adapter weight differences. The development data is separated from the selected training data, and the final 32-question comparison excludes the earlier eight-question diagnostic set. The official GSM8K test set is unused.

Compatibility work included CUDA 13 support in bitsandbytes, serialization of dataset columns, and a workaround for TRL 0.22.2 copying an initialized GRPO configuration during quantized-model preparation. Adapter gradients were explicitly restored after that preparation. CPU tests cover reward parsing, logging, and resume behavior.

## Diagnosis

Early five-step runs produced only one nonzero-gradient step. Response logs showed a mixture of wrong math, correct answers without tags, and identical rewards within a response group. GRPO needs differing rewards within a group to obtain a nonzero relative advantage; giving two equally correct responses credit would still produce equal rewards.

A four-response, 20-step pilot produced reward variation and nonzero gradients in 14 steps. All 224 adapter tensors changed. This established that optimization was active, not that mathematical ability improved. It is not a controlled comparison of generation counts because other run properties and sampled outputs differed.

## Evaluation mistake and correction

The historical training scorer (`v2-answer-field`) rejected correct tagged answers followed by a period or containing a unit. This accidentally penalized presentation. The training history remains unchanged and is explicitly labeled as having used that scorer.

Before evaluating the final adapter, a separate corrected evaluation scorer (`v3-tagged-numeric-presentation`) was frozen. It accepts one numeric answer field with bounded presentation normalization, while rejecting ambiguous or multiple answers. It does not hunt for the reference number in intermediate work. Rescoring the fixed training outputs increased accepted answers from 25 to 38 out of 80; that was a correction to measurement, not a model gain.

## Observed outcome

On 32 development questions, tagged-answer accuracy was 21/32 for the base and 23/32 for the adapter. Both newly accepted answers were already numerically correct in the base output. The adapter added answer tags. Only three response texts changed overall, and no evaluation output was truncated.

The defensible conclusion is a small observed improvement in answer-field compliance. There is no demonstrated gain in mathematical reasoning. This evaluation has one seed, a small sample and a policy chosen during development; it is not an official benchmark result.

## Engineering value

The resulting artifact supports inspection of the complete recorded trajectory, independent rescoring, verification of model/artifact fingerprints, and paired adapter-on/adapter-off replay. The saved adapter and tokenizer are bundled and their fingerprints match the recorded evaluation. The base model is downloaded separately. A reviewer can run the CPU audit without downloading a model.

## Next research step, if pursued

Use a corrected reward policy fixed before training, an explicit compute budget, multiple seeds where affordable, and separate evaluation of mathematical answers versus structural compliance. Keep ambiguous dataset labels documented. Do not choose metrics after seeing the test results. This is future work, not a completed claim.
