# SmolLM2 Function-Calling Fine-Tune (Educational)

> ⚠️ **This is an educational project.** It exists to learn how QLoRA
> fine-tuning, multi-turn chat templates, assistant-only loss and
> function-calling evaluation work in practice. It is **not**
> production-ready, not rigorously benchmarked, and a 1.7B model will
> happily emit wrong function calls. Use it to learn, not in production.

## What it does

Fine-tunes `HuggingFaceTB/SmolLM2-1.7B-Instruct` with **QLoRA**
(4-bit NF4 base + LoRA r=32) on the `glaiveai/glaive-function-calling-v2`
dataset, then measures whether the model:

1. generates **valid JSON** inside `<functioncall> {...} </functioncall>` blocks,
2. calls a tool **only when appropriate** (tool-vs-text intent accuracy),
3. selects the **correct tool name** (tool selection accuracy).

## Project layout

```
smollm2-function-calling-fine-tuning/
├── README.md               # this file
├── .gitignore              # ignores checkpoints, caches, venvs, ...
├── setup.py                # package setup: dependencies, resolving packages conflicts 
├── pyproject.toml          # minimal build-system shim for setup.py
├── requirements.txt        # flat dependency list (mirrors setup.py)
├── scripts/
│   └── setup_env.py        # env setup: pip install -e . + removes torchao
└── src/
    ├── __init__.py
    ├── config.py           # ALL configuration as Python dataclasses (no YAML)
    ├── data.py             # dataset loading, parsing, train/val/test splits
    ├── evaluation.py       # <functioncall> parsing + evaluate_function_calling
    ├── utils.py            # patched chat template (+ `tool` role), tokenizer
    ├── train.py            # QLoRA training entrypoint
    ├── merge_lora.py       # merge LoRA into a standalone model
    └── inference.py        # demo using the adapter published on the HF Hub
```

## Quickstart

```bash
cd smollm2-function-calling-fine-tuning

# 1) Install dependencies via setup.py (PyTorch must match your CUDA build!)
python scripts/setup_env.py t

# 2) Smoke test (a few steps, small sample)
python -m src.train --max-steps 20 --sample-size 2000 --skip-baseline

# 3) Full training run (~800 steps, needs a ~16GB GPU)
python -m src.train

# 4) Merge the LoRA adapter into a standalone model
python -m src.merge_lora --adapter ./smollm2-glaive-fc-final --output ./smollm2-glaive-fc-merged

# 5) Play with the adapter published on the Hugging Face Hub
python -m src.inference
```

> `scripts/setup_env.py` exists to solve dependency conflicts i got during experiment 
> lines are IPython magic and are a **syntax error** in plain `.py` files.

## Using the published adapter

The fine-tuned adapter is available on the Hugging Face Hub as
`SohaibAbdoAhmed/smollm2-glaive-fc-best-adapter`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = PeftModel.from_pretrained(base_model, "SohaibAbdoAhmed/smollm2-glaive-fc-best-adapter")
```

`src/inference.py` wraps this into a ready-to-run demo, including the patched
chat template (with the extra `tool` role) that the adapter was trained with.

## Configuration (dataclasses, no YAML)

Everything lives in `src/config.py` as nested dataclasses with defaults —
`Dataset`, `Training`, `Lora`, `Evaluation`, `Paths`, grouped under
`ExperimentConfig`:

```python
from src.config import load_config
cfg = load_config()          # -> ExperimentConfig
cfg.training.max_steps       # 800
cfg.lora.r                   # 32
```

CLI flags on `src/train.py` override individual values at runtime:
`--max-steps`, `--sample-size`, `--learning-rate`, `--output-dir`,
`--skip-baseline`.

## Metrics

| Metric | Meaning |
| --- | --- |
| Generated Valid JSON for Tools | of the outputs that attempted a tool call, how many were parseable JSON |
| Intent Accuracy (Tool vs Text) | did the model decide to call a tool exactly when the reference answer did |
| Correct Tool Selected | of the reference tool calls, how many used the right tool name |

## Hardware notes

* Single GPU with ~16 GB VRAM is enough.
* 4-bit NF4 + double quantization via `bitsandbytes`.
* Effective batch size: 2 x 16 gradient accumulation = 32.
* `assistant_only_loss=True`: loss is computed only on assistant tokens
  (requires the `{% generation %}` tags in the chat template).

## Credits

* Base model: HuggingFaceTB/SmolLM2-1.7B-Instruct
* Dataset: glaiveai/glaive-function-calling-v2
* Libraries: transformers, peft, trl, bitsandbytes, datasets, accelerate

---

*Educational example — provided as-is, no warranty, not affiliated with any
of the above organizations.*
