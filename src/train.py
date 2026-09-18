"""QLoRA fine-tuning of SmolLM2-1.7B-Instruct on glaive-function-calling-v2.

Pipeline (mirrors the original Colab notebook):
    1. Load + parse the multi-turn glaive dataset (train/val/test splits).
    2. Load the base model in 4-bit NF4.
    3. Run a PRE-TRAINING baseline evaluation on the blind test set.
    4. Fine-tune with LoRA + assistant-only loss (eval on the val set).
    5. Run the POST-TRAINING evaluation on the same blind test set.
    6. Merge the LoRA adapter into a standalone model.

All configuration comes from the dataclasses in src/config.py; CLI flags
override individual values.

Run from the repository root:
    python -m src.train
    python -m src.train --max-steps 20 --sample-size 2000 --skip-baseline   # smoke test
"""

import argparse

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .config import ExperimentConfig, load_config
from .data import load_and_prepare_datasets
from .evaluation import evaluate_function_calling
from .merge_lora import merge_and_save_lora
from .utils import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune SmolLM2 on glaive function-calling data")
    parser.add_argument("--max-steps", type=int, default=None, help="Override Training.max_steps")
    parser.add_argument("--sample-size", type=int, default=None, help="Override Dataset.sample_size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override Training.learning_rate")
    parser.add_argument("--output-dir", type=str, default=None, help="Override Paths.output_dir")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip the pre-training baseline evaluation")
    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Apply CLI overrides on top of the dataclass defaults."""
    if args.max_steps is not None:
        cfg.training.max_steps = args.max_steps
    if args.sample_size is not None:
        cfg.dataset.sample_size = args.sample_size
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir
    return cfg


def build_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(), args)

    print("Experiment config:\n" + cfg.summary())

    # ---------------------------------------------------------
    # 1. DATASET LOADING & MULTI-TURN PREPROCESSING
    # ---------------------------------------------------------
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(cfg)

    # ---------------------------------------------------------
    # 2. MODEL & TOKENIZER INITIALIZATION
    # ---------------------------------------------------------
    tokenizer = load_tokenizer(cfg.model_id, padding_side="right")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=build_quantization_config(),
        device_map="auto",
        dtype=torch.bfloat16,
    )

    # ---------------------------------------------------------
    # 3. PRE-TRAINING BASELINE EVALUATION (ON TEST SET)
    # ---------------------------------------------------------
    if not args.skip_baseline:
        model.config.use_cache = True
        evaluate_function_calling(
            model,
            tokenizer,
            test_dataset,
            batch_size=cfg.evaluation.eval_batch_size,
            n_samples=cfg.evaluation.max_eval_samples,
            eval_name="PRE-TRAINING BASELINE",
            max_new_tokens=cfg.evaluation.max_new_tokens,
        )

    # Prepare model state for kbit fine-tuning
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # ---------------------------------------------------------
    # 4. TRAINING CONFIGURATION & EXECUTION
    # ---------------------------------------------------------
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type="CAUSAL_LM",
        target_modules=cfg.lora.target_modules,
    )

    sft_config = SFTConfig(
        output_dir=cfg.paths.output_dir,
        max_steps=cfg.training.max_steps,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.training.eval_steps,
        save_strategy="steps",
        save_steps=cfg.training.save_steps,
        bf16=cfg.training.bf16,
        optim=cfg.training.optim,
        report_to="none",
        max_length=cfg.training.max_length,
        packing=False,
        assistant_only_loss=True,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # validation set for eval / best-checkpoint tracking
        processing_class=tokenizer,
    )

    print("\nStarting Fine-tuning...")
    trainer.train()
    trainer.save_model(cfg.paths.final_dir)
    tokenizer.save_pretrained(cfg.paths.final_dir)

    # ---------------------------------------------------------
    # 5. POST-TRAINING EVALUATION (ON TEST SET)
    # ---------------------------------------------------------
    del trainer, model
    torch.cuda.empty_cache()

    eval_model = AutoPeftModelForCausalLM.from_pretrained(
        cfg.paths.final_dir,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    eval_tokenizer = load_tokenizer(cfg.paths.final_dir)

    evaluate_function_calling(
        eval_model,
        eval_tokenizer,
        test_dataset,
        batch_size=cfg.evaluation.eval_batch_size,
        n_samples=cfg.evaluation.max_eval_samples,
        eval_name="POST-TRAINING",
        max_new_tokens=cfg.evaluation.max_new_tokens,
    )

    del eval_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 6. MERGE WEIGHTS AND SAVE THE STANDALONE MODEL
    # ---------------------------------------------------------
    merge_and_save_lora(cfg.paths.final_dir, cfg.paths.merged_dir)


if __name__ == "__main__":
    main()
