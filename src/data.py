"""Dataset loading & multi-turn preprocessing (glaiveai/glaive-function-calling-v2)."""

import re

from datasets import Dataset, load_dataset
from tqdm import tqdm

from .config import ExperimentConfig

ROLE_MAP = {"USER:": "user", "ASSISTANT:": "assistant", "FUNCTION RESPONSE:": "tool"}


def parse_glaive_chat(row):
    """Turn one raw glaive row into a list of chat messages.

    The raw `chat` column looks like:
        USER: ...\nASSISTANT: ...\n<functioncall> {...}\nFUNCTION RESPONSE: ...
    """
    system_msg = row["system"].replace("SYSTEM: ", "").strip()
    chat = row["chat"]

    messages = [{"role": "system", "content": system_msg}]
    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", chat)

    current_role, current_content = None, []

    for p in parts:
        p = p.strip()
        if p in ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:"):
            if current_role:
                messages.append(
                    {"role": ROLE_MAP[current_role], "content": "\n".join(current_content).strip()}
                )
            current_role, current_content = p, []
        elif p:
            current_content.append(p)

    if current_role:
        messages.append(
            {"role": ROLE_MAP[current_role], "content": "\n".join(current_content).strip()}
        )

    return messages if len(messages) > 1 else None


def _is_well_formed(messages):
    """Keep only conversations we can safely evaluate: system, user, assistant, ...

    (Guards the evaluation loop, which assumes messages[1] is a user turn and
    messages[2] the first assistant answer.)
    """
    return (
        messages is not None
        and len(messages) >= 3
        and messages[1]["role"] == "user"
        and messages[2]["role"] == "assistant"
    )


def load_and_prepare_datasets(cfg: ExperimentConfig):
    """Load, parse and split the glaive dataset into train / val / test."""
    dataset = load_dataset(cfg.dataset.name)
    df = dataset["train"].to_pandas()

    # Subsample for fast iteration (raise sample_size for longer runs)
    df = df.sample(cfg.dataset.sample_size, random_state=cfg.dataset.seed).reset_index(drop=True)

    tqdm.pandas(desc="Formatting multi-turn dataset")
    df["messages"] = df.progress_apply(parse_glaive_chat, axis=1)
    finetune_df = df.dropna(subset=["messages"])[["messages"]].reset_index(drop=True)

    # Drop malformed conversations (see _is_well_formed)
    finetune_df = finetune_df[finetune_df["messages"].apply(_is_well_formed)].reset_index(drop=True)

    hf_dataset = Dataset.from_pandas(finetune_df)

    # 90% train / 5% validation (early stopping) / 5% test (blind eval)
    train_test = hf_dataset.train_test_split(test_size=cfg.dataset.test_size, seed=cfg.dataset.seed)
    val_test = train_test["test"].train_test_split(
        test_size=cfg.dataset.val_fraction_of_test, seed=cfg.dataset.seed
    )

    train_dataset = train_test["train"]
    val_dataset = val_test["train"]
    test_dataset = val_test["test"]

    print(f"Dataset Splits -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset
