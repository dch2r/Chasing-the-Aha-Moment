# Reproducibility boundaries

## Included and verified

The repository contains the saved LoRA adapter and tokenizer in `adapter/`, plus the original `config.json`, `trainer_config.json`, `environment.txt`, model revision, data manifest, 128 selected training records, eight prior development records, training utility sources and per-tensor adapter change log in `results/pilot/run_metadata/`. The adapter weights, adapter configuration and tokenizer SHA-256 fingerprints match the frozen evaluation protocol. Original metadata bytes are preserved.

The CPU verifier checks all 128 selected training IDs and eight prior development IDs against the 32 evaluation IDs, not only the 20 training questions observed in response logs. It also checks the saved dataset fingerprint, model revision, core configuration and training utility sources. Recorded evaluation and training results are never modified by the scripts.

## External requirements and excluded files

The pinned Qwen2.5-1.5B-Instruct base model and runtime dependencies must be downloaded for GPU execution. Intermediate checkpoints, optimizer state and `training_args.bin` are not included; they are unnecessary for saved-adapter inference. The repository therefore cannot resume the exact optimizer state. The original historical Llama report is also not included.

The portable notebook uses the bundled adapter. The recorded evaluation notebook in `notebooks/recorded/` keeps its original Colab Drive paths for provenance. The fixed adapter is tracked through a specific `.gitignore` exception; new checkpoint and output directories are ignored.

## Environment

The original environment listing is included. The evaluation protocol records PyTorch 2.11.0+cu130, Transformers 4.56.2, PEFT 0.17.1, datasets 4.1.1, accelerate 1.10.1 and bitsandbytes 0.48.1. The pilot used TRL 0.22.2. `requirements.txt` pins direct packages; it is not a complete dependency lockfile. Use a compatible CUDA-enabled PyTorch build.

## Verification versus reproduction

The CPU checks validate stored records and artifact identity. The portable evaluator was run on a Colab Tesla T4, and its complete outputs are stored in `results/replay/`. All 64 response texts match the original evaluation exactly, and pairing, hashes, per-response scores and summary counts were checked from those files. No new training was performed. A replay may produce different outputs across hardware or runtime versions and is not a new independent evaluation. The evidence supports a formatting improvement on two development examples, not a demonstrated reasoning improvement.
