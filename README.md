# Chasing the Aha Moment

**Auditing reward design and evaluation in GRPO post-training.**

A short Qwen2.5-1.5B GRPO run raised tagged-answer accuracy from 21/32 to 23/32 on development questions. Inspecting the changed answers showed that both gains came from adding answer tags to numbers the base model already calculated correctly. This project investigates that gap between a higher score and a better answer.

## What is implemented

- Four-bit NF4 training with rank-16 LoRA on a single Colab T4.
- Per-response reward logs, gradient diagnostics and adapter-weight change checks.
- Separate training and evaluation parsers, with tests for ambiguous tags, punctuation and numeric normalization.
- Paired base/adapter evaluation with fixed prompts, decoding and model revision.
- Resumable evaluation, file fingerprints and a CPU audit of saved evidence.
- Paired statistical analysis and exports of every changed response for inspection.

The training algorithm comes from TRL. The historical Llama notebooks build on Unsloth examples. The contribution here is the instrumentation, evaluation workflow and analysis, rather than a new RL algorithm.

## Results

| Measure | Base | Trained adapter |
| --- | ---: | ---: |
| Correct tagged answers | 21/32 | 23/32 |
| Parseable answers | 23/32 | 25/32 |
| Strict reasoning/answer format | 1/32 | 2/32 |
| Truncated outputs | 0/32 | 0/32 |

The 20-step pilot sampled 80 responses. Fourteen steps had nonzero gradients and all 224 LoRA tensors changed. Training and saving took 794 seconds, with 3.21 GiB peak PyTorch allocated memory.

Only three response texts changed in the paired evaluation. The two newly accepted answers were formatting changes. The paired bootstrap 95% interval for the tagged-score difference is 0 to 15.625 percentage points; the exact paired McNemar test gives p = 0.50. These are exploratory development results, not evidence of improved mathematical reasoning or an official GSM8K test score.

A separate run of the portable evaluator on a Tesla T4 reproduced **all 64 response texts exactly**. Saved responses, configurations, adapter and tokenizer are included.

## Quick start: inspect without a GPU

From the repository directory, with Python 3.10 or newer:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python scripts/verify_results.py
python scripts/analyze_results.py
```

These commands require only the Python standard library. To export the analysis and changed responses, use `python scripts/analyze_results.py --output runs/analysis` with a new output directory.

## Replay on a GPU

The saved adapter is included in `adapter/` and tracked by Git. The base model is downloaded from Hugging Face. Install a compatible CUDA-enabled PyTorch, then:

```bash
python -m pip install -r requirements.txt
python scripts/check_adapter.py --adapter adapter
python scripts/evaluate.py --adapter adapter --output runs/replay
```

The tested environment is recorded in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). The evaluator uses a 20-minute soft generation budget. If it prints `PAUSED`, rerun the same command to continue. Outputs cannot overwrite the saved evidence or adapter.

For Colab, open [notebooks/Evaluate_Saved_Adapter.ipynb](notebooks/Evaluate_Saved_Adapter.ipynb) in a T4 session. Run installation, restart, then run the remaining cells. Set `REPO_URL` to this repository's URL to clone it, or leave it empty and upload the repository ZIP when prompted. Drive is used only to preserve replay outputs. No retraining is needed.

## Training and further experiments

[notebooks/Train_Pilot.ipynb](notebooks/Train_Pilot.ipynb) contains the recorded 20-step workflow. It intentionally preserves the historical reward parser, including its known presentation-related false negatives.

The next study compares historical, corrected and format-only rewards under matched settings. The protocol and tested reward components are prepared in [experiments/](experiments/); **those training runs have not been performed**. See [docs/NEXT_EXPERIMENT.md](docs/NEXT_EXPERIMENT.md) before allocating GPU time.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/`, `scripts/`, `tests/` | Scoring, diagnostics, evaluation, analysis and validation |
| `notebooks/` | Training and portable evaluation entry points |
| `adapter/` | Saved LoRA adapter and tokenizer |
| `results/pilot/` | Training responses, metrics and original run metadata |
| `results/evaluation/` | Original 32-question paired evaluation |
| `results/replay/` | Completed portable replay |
| `results/analysis/` | Recomputed statistics and changed-answer evidence |
| `legacy/original/` | Historical Llama 8B notebooks |

The historical notebook `legacy/original/Llama3_1_(8B)_GRPO_250.ipynb` is not the baseline for the Qwen experiment. Its limitations and relation to this work are explained in [docs/LLAMA_REVIEW.md](docs/LLAMA_REVIEW.md). The actual baseline is the same Qwen model with its adapter disabled.

## Scope and attribution

This is one short training run and a small development evaluation. The official GSM8K test split was not used. Mathematical correctness and format compliance require separate treatment; corrected offline scoring is not a model improvement.

Project owner: **Diwesh Chaurasia**. Development used AI assistance. Upstream notebooks, models and data retain their attribution and license terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Project-owned additions use the MIT license.
