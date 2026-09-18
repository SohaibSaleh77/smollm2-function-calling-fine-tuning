"""Merge a trained LoRA adapter into the base model and save a standalone model.

Run from the repository root:
    python -m src.merge_lora --adapter ./smollm2-glaive-fc-final --output ./smollm2-glaive-fc-merged
"""

import argparse

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def merge_and_save_lora(peft_model_path: str, save_path: str) -> None:
    print(f"\nMerging LoRA weights from {peft_model_path} into base model...")
    base_model = AutoPeftModelForCausalLM.from_pretrained(
        peft_model_path,
        device_map="cpu",
        dtype=torch.bfloat16,
    )
    merged_model = base_model.merge_and_unload()
    merged_model.save_pretrained(save_path, safe_serialization=True, max_shard_size="2GB")

    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    tokenizer.save_pretrained(save_path)

    print(f"Merged standalone model successfully saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model")
    parser.add_argument("--adapter", required=True, help="Path to the saved PEFT adapter (e.g. ./smollm2-glaive-fc-final)")
    parser.add_argument("--output", required=True, help="Where to save the merged standalone model")
    args = parser.parse_args()
    merge_and_save_lora(args.adapter, args.output)


if __name__ == "__main__":
    main()
