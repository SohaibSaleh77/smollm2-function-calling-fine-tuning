"""Experiment configuration as pure Python dataclasses.



from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Dataset:
    """Dataset-related settings.

    Note: this class lives only in this module and is consumed as
    `cfg.dataset`; it is never imported alongside `datasets.Dataset`,
    so the shared name is safe.
    """

    name: str = "glaiveai/glaive-function-calling-v2"
    sample_size: int = 15000          # subsample for fast iteration
    seed: int = 42
    test_size: float = 0.10           # 10% held out -> 5% val + 5% test
    val_fraction_of_test: float = 0.50


@dataclass
class Training:
    max_steps: int = 800
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16   # effective batch size = 32
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 50
    logging_steps: int = 50
    eval_steps: int = 100
    save_steps: int = 100
    max_length: int = 1536
    bf16: bool = True
    optim: str = "paged_adamw_8bit"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True


@dataclass
class Lora:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: str = "all-linear"


@dataclass
class Evaluation:
    max_eval_samples: int = 200
    eval_batch_size: int = 4
    max_new_tokens: int = 256


@dataclass
class Paths:
    output_dir: str = "./smollm2-glaive-fc"         # checkpoints during training
    final_dir: str = "./smollm2-glaive-fc-final"    # best adapter after training
    merged_dir: str = "./smollm2-glaive-fc-merged"  # standalone merged model


@dataclass
class ExperimentConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    hub_adapter_id: str = "SohaibAbdoAhmed/smollm2-glaive-fc-best-adapter"
    dataset: Dataset = field(default_factory=Dataset)
    training: Training = field(default_factory=Training)
    lora: Lora = field(default_factory=Lora)
    evaluation: Evaluation = field(default_factory=Evaluation)
    paths: Paths = field(default_factory=Paths)

    def summary(self) -> str:
        """Human-readable dump of the whole config."""
        return json.dumps(asdict(self), indent=2)


def load_config() -> ExperimentConfig:
    """Build the default configuration (pure dataclasses, no YAML)."""
    return ExperimentConfig()
