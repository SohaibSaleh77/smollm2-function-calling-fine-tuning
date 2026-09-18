"""Evaluation & parsing utilities for function-calling behaviour."""

import ast
import json
import re

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def parse_function_call(text: str) -> dict:
    """Classify a model output as either a plain text reply or a function call."""
    fc_match = re.search(r"<functioncall>\s*({.*})", text, re.DOTALL)
    if fc_match:
        json_str = fc_match.group(1).strip()
        try:
            # 1. Try strict JSON first
            data = json.loads(json_str)
            return {"type": "function", "name": data.get("name", "UNKNOWN"), "is_valid_json": True}
        except json.JSONDecodeError:
            try:
                # 2. Fallback to safe AST eval (handles Python dicts with single quotes)
                data = ast.literal_eval(json_str)
                if isinstance(data, dict):
                    return {"type": "function", "name": data.get("name", "UNKNOWN"), "is_valid_json": True}
            except Exception:
                pass

        # 3. If both fail, it's invalid syntax
        return {"type": "function", "name": "INVALID_SYNTAX", "is_valid_json": False}
    else:
        return {"type": "text", "name": None, "is_valid_json": True}


def evaluate_function_calling(
    model,
    tokenizer,
    eval_dataset,
    batch_size: int = 4,
    n_samples: int = 200,
    eval_name: str = "EVALUATION",
    max_new_tokens: int = 256,
):
    """Batched evaluation: JSON validity, tool-vs-text intent accuracy, tool selection."""
    model.eval()

    # Text generation requires left-padding during batching
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    true_intents, pred_intents = [], []
    true_names, pred_names = [], []
    valid_json_count = 0
    total_fc_count = 0

    n_samples = min(n_samples, len(eval_dataset))

    for i in tqdm(range(0, n_samples, batch_size), desc=f"Running {eval_name}"):
        batch = eval_dataset[i : i + batch_size]

        # We test how it handles the initial request (System + User prompt)
        batch_messages = [row[:2] for row in batch["messages"]]
        batch_truths = [row[2]["content"] for row in batch["messages"]]

        inputs = tokenizer.apply_chat_template(
            batch_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, prompt_len:]
        gen_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for gen_text, true_assistant_msg in zip(gen_texts, batch_truths):
            parsed_pred = parse_function_call(gen_text)
            parsed_true = parse_function_call(true_assistant_msg)

            true_intents.append(parsed_true["type"])
            pred_intents.append(parsed_pred["type"])

            if parsed_true["type"] == "function":
                true_names.append(parsed_true["name"])
                pred_names.append(parsed_pred["name"] if parsed_pred["type"] == "function" else "FAILED_TO_CALL")

            if parsed_pred["type"] == "function":
                total_fc_count += 1
                if parsed_pred["is_valid_json"]:
                    valid_json_count += 1

    tokenizer.padding_side = original_padding_side

    print("\n" + "=" * 55)
    print(f" 🛠️ {eval_name} REPORT")
    print("=" * 55)

    intent_acc = accuracy_score(true_intents, pred_intents) * 100
    name_acc = accuracy_score(true_names, pred_names) * 100 if true_names else 0.0
    json_validity = (valid_json_count / total_fc_count * 100) if total_fc_count > 0 else 0.0

    print("\n1. SYNTAX & FORMATTING:")
    print(f"   Generated Valid JSON for Tools: {valid_json_count}/{total_fc_count} ({json_validity:.2f}%)")

    print("\n2. TOOL LOGIC PERFORMANCE:")
    print(f"   Intent Accuracy (Tool vs Text): {intent_acc:.2f}%")
    print(f"   Correct Tool Selected:          {name_acc:.2f}%")
    print("=" * 55 + "\n")
