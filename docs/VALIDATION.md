# Validation

- 67 CPU unit tests cover reward extraction, logging, interrupted-output recovery, adapter fingerprint checks, paired analysis, the reward-ablation components and the ablation logging and summary.
- `scripts/verify_results.py` checks 64 evaluation outputs, 80 training responses, protocol and scorer hashes, full 128-training / 8-prior-development / 32-evaluation ID separation and consistency between standalone modules and their notebook-embedded copies.
- All Python files and executable notebook cells parse. The portable command-line help works without importing GPU dependencies.
- Historical notebook code and outputs are unchanged; see `legacy/README.md` for the metadata edits.

The original pilot, evaluation and portable replay were run on Colab. The replay in `results/replay/` contains 64 unique paired responses, all text-identical to the originals, with per-response scores and summary counts checked independently. The replay protocol records a Tesla T4 and the same six package versions as the original evaluation, and its protocol and adapter fingerprints match the originals.

A second replay on 2026-10-30 used the notebook's GitHub clone path instead of an uploaded ZIP. It also paused at the budget (62 of 64) and completed on resumption, and all 64 response texts again matched the originals exactly, with the same 21/32 and 23/32 scores.

The first replay paused at the 20-minute budget with 60 of 64 responses saved and completed on resumption; the final file has 64 unique responses and no duplicates. The GPU evaluator was later reformatted for readability; its Python AST matches the T4-tested version. Frozen scoring and logging modules remain byte-identical.

The prospective reward components and split selection pass CPU tests; no reward-ablation training has been run. The exact paired McNemar p-value is 0.50 and the bootstrap interval matches the reported 0 to 15.625 percentage points.
