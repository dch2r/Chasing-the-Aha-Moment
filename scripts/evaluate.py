"""Portable, resumable inference only. Never trains or writes to adapter files."""

import argparse, json, sys, time, hashlib, math, random, importlib.metadata
from pathlib import Path
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from artifact_checks import verify_adapter, sha256
from eval_scoring import score, VERSION
from evaluation_io import read_rows, append_row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Local directory containing saved adapter AND tokenizer files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New/resumable results directory; never the recorded results directory",
    )
    parser.add_argument("--budget-seconds", type=int, default=1200)
    a = parser.parse_args()
    if a.budget_seconds <= 0:
        parser.error("budget-seconds must be positive")
    output = a.output.resolve()
    for protected in [a.adapter.resolve(), (ROOT / "results").resolve()]:
        if output == protected or protected in output.parents:
            parser.error(
                "Output must be outside adapter and recorded results directories"
            )
    protocol = json.loads((ROOT / "results/evaluation/protocol.json").read_text())
    verify_adapter(a.adapter, protocol)
    if sha256(ROOT / "src/eval_scoring.py") != protocol["scorer_sha256"]:
        parser.error("Scorer changed from the frozen protocol")
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        GenerationConfig,
        set_seed,
    )
    from peft import PeftModel, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        parser.error(
            "CUDA GPU required. Use verify_results.py for CPU-only verification."
        )
    versions = {k: importlib.metadata.version(k) for k in protocol["versions"]}
    spec = dict(
        recorded_protocol_sha256=sha256(ROOT / "results/evaluation/protocol.json"),
        adapter_sha256=protocol["adapter_sha256"],
        scorer_version=VERSION,
        environment=versions,
        gpu=torch.cuda.get_device_name(0),
        purpose="Replay of the same recorded 32-question protocol; not fresh independent evidence",
    )
    text = json.dumps(spec, sort_keys=True, indent=2)
    run_hash = hashlib.sha256(text.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    header = output / "replay_protocol.json"
    if header.exists() and header.read_text() != text:
        parser.error(
            "Existing output has a different environment/protocol. Choose a new output directory."
        )
    if not header.exists():
        header.write_text(text)
    questions = protocol["questions"]
    ids = {q["source_id"] for q in questions}
    logfile = output / "evaluation_predictions.jsonl"
    rows = read_rows(logfile, ids)
    for row in rows.values():
        if row.get("protocol_hash") != run_hash:
            parser.error("Saved prediction belongs to a different replay protocol")
    if len(rows) < 2 * len(questions):
        set_seed(protocol["seed"])
        tokenizer = AutoTokenizer.from_pretrained(str(a.adapter))
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            protocol["model"],
            revision=protocol["revision"],
            quantization_config=quant,
            device_map={"": 0},
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
        model = PeftModel.from_pretrained(base, str(a.adapter), is_trainable=False)
        model.eval()
        model.config.use_cache = True
        for param in model.parameters():
            param.requires_grad_(False)
        assert model.get_model_status().enabled is True
        config = GenerationConfig(
            do_sample=False,
            max_new_tokens=protocol["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
        )
        started = time.perf_counter()
        for i, q in enumerate(questions):
            if all((label, q["source_id"]) in rows for label in ("base", "tuned")):
                continue
            if time.perf_counter() - started >= a.budget_seconds:
                break
            prompt = tokenizer.apply_chat_template(
                q["prompt"], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            for label in ["base", "tuned"] if i % 2 == 0 else ["tuned", "base"]:
                key = (label, q["source_id"])
                if key in rows:
                    continue
                context = model.disable_adapter() if label == "base" else nullcontext()
                torch.cuda.synchronize()
                t = time.perf_counter()
                with context, torch.inference_mode():
                    out = model.generate(**inputs, generation_config=config)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                tokens = out[0, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(tokens, skip_special_tokens=True)
                row = dict(
                    label=label,
                    source_id=q["source_id"],
                    gold=q["gold"],
                    response=response,
                    generated_tokens=len(tokens),
                    hit_token_cap=len(tokens) >= protocol["max_new_tokens"],
                    seconds=elapsed,
                    protocol_hash=run_hash,
                    **score(response, q["gold"]),
                )
                append_row(logfile, row)
                rows[key] = row
                print(f"{len(rows)}/{2*len(questions)} answers saved", flush=True)
    result = {
        "complete": len(rows) == 2 * len(questions),
        "saved_answers": len(rows),
        "expected_questions": len(questions),
        "protocol_hash": run_hash,
        "note": spec["purpose"],
    }
    if result["complete"]:
        for label in ["base", "tuned"]:
            selected = [rows[label, q["source_id"]] for q in questions]
            result[label] = {
                k: sum(r[k] for r in selected)
                for k in [
                    "correct",
                    "answer_parseable",
                    "format_valid",
                    "hit_token_cap",
                ]
            }
            result[label]["accuracy"] = result[label]["correct"] / len(questions)
        ds = [
            int(rows["tuned", q["source_id"]]["correct"])
            - int(rows["base", q["source_id"]]["correct"])
            for q in questions
        ]
        result.update(
            changed_to_correct=ds.count(1),
            changed_to_wrong=ds.count(-1),
            accuracy_difference_percentage_points=100 * sum(ds) / len(ds),
        )
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2))
    print(
        "COMPLETE"
        if result["complete"]
        else "PAUSED — rerun the same command to resume"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
