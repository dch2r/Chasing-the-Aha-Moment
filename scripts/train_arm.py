"""Train one fresh reward-ablation adapter (one arm, one seed) on a single GPU.

Mirrors notebooks/Train_Pilot.ipynb: same 128 training records, prompt, model
revision, LoRA, GRPO settings and compatibility fixes. Only the correctness
reward changes between arms. Never touches the recorded results.
"""

import argparse, json, math, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from experiment_rewards import ARMS
from rewards import SYSTEM_PROMPT, format_reward

PLAN = ROOT / "experiments/reward_ablation.json"
TRAIN_RECORDS = ROOT / "results/pilot/run_metadata/train.jsonl"


def stop_reason(steps_done, steps_wanted, budget_hit):
    if steps_done >= steps_wanted:
        return "step_limit"
    return "time_budget" if budget_hit else "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--budget-seconds", type=int, default=1800)
    a = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    if a.seed not in plan["seeds"]:
        parser.error(f"Seed must be one of the planned seeds {plan['seeds']}")
    if a.steps not in (plan["feasibility_steps"], plan["main_steps"]):
        parser.error("Steps must match the planned feasibility or main budget")
    run = a.output.resolve()
    if run == ROOT or ROOT in run.parents:
        parser.error("Output must be outside the repository")
    run.mkdir(parents=True, exist_ok=False)

    records = [json.loads(line) for line in TRAIN_RECORDS.read_text().splitlines()]
    assert [r["source_id"] for r in records] == plan["train_ids"], "Train IDs differ"
    assert all(r["prompt"][0]["content"] == SYSTEM_PROMPT for r in records)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import GRPOTrainer
    from arm_logging import make_arm_logged_reward
    from diagnostics import summarize
    from grpo_compat import CompatibleGRPOConfig, TrainingTimeBudget

    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    cfg = dict(
        arm=a.arm,
        seed=a.seed,
        steps=a.steps,
        model=plan["model"],
        revision=plan["revision"],
        train_ids_from="results/pilot/run_metadata/train.jsonl",
        num_generations=plan["num_generations"],
        max_prompt_length=plan["max_prompt_length"],
        max_completion_length=plan["max_completion_length"],
        rank=plan["lora_rank"],
        learning_rate=plan["learning_rate"],
        training_budget_seconds=a.budget_seconds,
        gpu=torch.cuda.get_device_name(0),
    )
    (run / "config.json").write_text(json.dumps(cfg, indent=2))

    set_seed(a.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train = Dataset.from_list(records)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        revision=cfg["revision"],
        quantization_config=quant,
        device_map={"": 0},
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["rank"],
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    resolved = getattr(model.config, "_commit_hash", None)
    assert resolved in (None, cfg["revision"]), f"Unexpected model commit {resolved}"

    budget = TrainingTimeBudget(a.budget_seconds)
    logged = make_arm_logged_reward(
        run / "training_responses.jsonl", a.arm, cfg["max_completion_length"]
    )
    args = CompatibleGRPOConfig(
        output_dir=str(run / "checkpoints"),
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["num_generations"],
        num_generations=cfg["num_generations"],
        max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=cfg["max_completion_length"],
        max_steps=a.steps,
        save_strategy="no",
        logging_steps=1,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        seed=a.seed,
        data_seed=a.seed,
        use_vllm=False,
        beta=0.0,
        temperature=0.8,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        reward_funcs=[logged, format_reward],
        train_dataset=train,
        callbacks=[budget],
    )
    # TRL re-prepares the quantized model and freezes everything; restore LoRA.
    for name, parameter in trainer.model.named_parameters():
        parameter.requires_grad_("lora_" in name)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    assert trainable > 0, "No trainable LoRA parameters found"
    trainer.model.train()
    (run / "trainer_config.json").write_text(trainer.args.to_json_string())
    (run / "environment.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
    )
    before = {
        name: p.detach().float().cpu().clone()
        for name, p in trainer.model.named_parameters()
        if p.requires_grad
    }

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train()
    seconds = time.perf_counter() - start
    trainer.save_model(str(run / "adapter"))
    steps_done = trainer.state.global_step
    reason = stop_reason(steps_done, a.steps, budget.triggered)
    (run / "training.json").write_text(
        json.dumps(
            dict(
                metrics=result.metrics,
                seconds=seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                history=trainer.state.log_history,
                requested_steps=a.steps,
                completed_steps=steps_done,
                stop_reason=reason,
            ),
            indent=2,
        )
    )

    per_tensor = []
    for name, p in trainer.model.named_parameters():
        if name in before:
            delta = p.detach().float().cpu() - before[name]
            assert torch.isfinite(delta).all().item(), f"Nonfinite change: {name}"
            per_tensor.append(
                dict(
                    name=name,
                    changed_elements=int(torch.count_nonzero(delta).item()),
                    squared_l2=float(delta.double().square().sum().item()),
                )
            )
    summary = summarize(run / "training_responses.jsonl")
    summary.update(
        arm=a.arm,
        seed=a.seed,
        requested_steps=a.steps,
        completed_steps=steps_done,
        stop_reason=reason,
        complete=reason == "step_limit",
        steps_with_nonzero_gradient=sum(
            r.get("grad_norm", 0) > 0 for r in trainer.state.log_history
        ),
        tensors_checked=len(per_tensor),
        tensors_changed=sum(x["changed_elements"] > 0 for x in per_tensor),
        l2_change=math.sqrt(sum(x["squared_l2"] for x in per_tensor)),
    )
    (run / "adapter_deltas.json").write_text(json.dumps(per_tensor, indent=2))
    (run / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("Run saved:", run)


if __name__ == "__main__":
    main()
