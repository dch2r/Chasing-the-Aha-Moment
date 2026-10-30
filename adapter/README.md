---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
license: apache-2.0
tags:
- grpo
- lora
- evaluation
---

# Qwen2.5-1.5B GRPO pilot adapter

A rank-16 LoRA adapter produced by Diwesh Chaurasia for the Chasing the Aha Moment reward-auditing study. This directory includes the adapter and the base model's tokenizer files. The base model weights are downloaded separately.

## Training

- Base: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Hardware: one Colab Tesla T4; NF4 quantization and FP16 compute.
- Method: TRL GRPO, 20 optimizer steps, four generations per question, learning rate 5e-6.
- LoRA: rank 16, alpha 32, attention q/k/v/o projections, zero dropout.
- Data: 128 selected GSM8K training examples; the run sampled 20 question groups and 80 responses.
- Rewards: historical numeric answer-field correctness (2.0) plus strict format compliance (0.5).
- Completion limit: 512 tokens. Training and saving took 794 seconds.

The historical correctness parser has known false negatives involving presentation. Its exact source and the original configuration are retained under `results/pilot/run_metadata/`.

## Evaluation and intended use

The paired development evaluation on 32 disjoint questions scored 21 correct tagged answers for the base and 23 for the adapter. Both newly accepted answers came from adding tags around already-correct numbers. Strict full-format compliance was 2/32. This adapter is intended for studying reward and evaluation behavior, not as a generally improved mathematics model.

A portable T4 replay reproduced all 64 response texts. No official GSM8K test result, multi-seed robustness or production suitability is established.

From the repository root:

```bash
python scripts/check_adapter.py --adapter adapter
python scripts/evaluate.py --adapter adapter --output runs/replay
```

The adapter inherits the base model's Apache-2.0 context; the license text is in `../LICENSES/Apache-2.0.txt`. Tokenizer assets originate from Qwen. See `../THIRD_PARTY_NOTICES.md` for source attribution. This model card replaces an automatically generated placeholder card; the adapter and tokenizer bytes are unchanged.
