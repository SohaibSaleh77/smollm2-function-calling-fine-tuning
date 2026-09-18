"""Inference demo for the fine-tuned function-calling model.

The trained LoRA adapter is published on the Hugging Face Hub, so you can use
it exactly like this:

    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    model = PeftModel.from_pretrained(base_model, "SohaibAbdoAhmed/smollm2-glaive-fc-best-adapter")

Run from the repository root:
    python -m src.inference                                      # Hub adapter
    python -m src.inference --adapter ./smollm2-glaive-fc-final  # local adapter
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .config import load_config
from .utils import load_tokenizer

CFG = load_config()  # dataclass config -> single source of truth

DEMO_SYSTEM_PROMPT = """You are a helpful assistant with access to the following functions. Use the functions if required:

{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use."
            }
        },
        "required": ["location", "format"]
    }
}"""

DEMO_USER_PROMPT = "What is the weather like in San Francisco?"


def load_function_calling_model(base_model_id: str, adapter_id: str):
    """Load the base model and attach the fine-tuned LoRA adapter."""
    tokenizer = load_tokenizer(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_id)
    model.eval()
    return model, tokenizer


def chat(model, tokenizer, messages, max_new_tokens: int = 256) -> str:
    """Generate a single assistant reply for a list of chat messages."""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo of the fine-tuned function-calling model")
    parser.add_argument("--adapter", default=CFG.hub_adapter_id, help="Hub repo id or local path of the LoRA adapter")
    args = parser.parse_args()

    model, tokenizer = load_function_calling_model(base_model_id=CFG.model_id, adapter_id=args.adapter)

    messages = [
        {"role": "system", "content": DEMO_SYSTEM_PROMPT},
        {"role": "user", "content": DEMO_USER_PROMPT},
    ]

    print("User:", DEMO_USER_PROMPT)
    reply = chat(model, tokenizer, messages)
    print("Assistant:", reply)


if __name__ == "__main__":
    main()
