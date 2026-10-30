# What the original Llama notebook establishes

The file remains at `legacy/original/Llama3_1_(8B)_GRPO_250.ipynb`.

## Evidence in the file

- The configured model is Unsloth's Llama 3.1 8B Instruct variant, quantized to four bits.
- Rank-32 LoRA targets attention and MLP projections; six responses are generated per prompt.
- Five correctness/format-related reward functions are used, including overlapping XML-format rewards.
- Saved output reports 250 optimizer steps and 13,743.20 seconds of training, about 3 hours 49 minutes. This is a historical log, not a forecast for a new run.
- The notebook trains on GSM8K's train split and saves a LoRA adapter to `grpo_saved_lora`.

## Why it is not the Qwen baseline

The current paired experiment compares Qwen2.5-1.5B with its adapter disabled versus enabled, using identical questions, prompt and decoding. Llama 8B is a different model, with different rewards, training settings and inference behavior. It is a historical starting point, not a controlled baseline.

The Llama demo compares a user-only “Calculate pi.” prompt with an adapter prompt that also includes the formatting system instruction. Both use sampling. This changes more than the adapter, so the example cannot isolate an adapter effect. There is no matched held-out GSM8K benchmark in this notebook to substantiate a before/after accuracy claim.

The current install cells pin Transformers 4.56.2 and TRL 0.22.2, while saved outputs show Transformers 4.48.3, PyTorch 2.5.1+cu124 and vLLM 0.7.3. Code and saved outputs therefore do not document one consistent executable environment.

## What the new work adds

The Qwen workflow replaces anecdotal demos with paired evaluation, records each sampled training response and reward, checks adapter changes, separates training/development examples, tests parsing failures and resumes interrupted evaluation. Its T4 replay reproduces all 64 original response texts. It does not validate the old Llama benchmark claim.

## Recommendation

Do not rerun the old notebook unchanged. First complete a controlled reward comparison on the smaller, validated Qwen pipeline. If the historical Llama adapter still exists, evaluate that saved adapter against its exact base model under matched prompts before considering retraining. A later Llama replication should use a fresh protocol, pinned environment and model revision, the same evaluation questions across arms, and separately report hardware cost. No such Llama replication has been run here.
