"""Shared utilities: the patched chat template used everywhere in the project."""

from transformers import AutoTokenizer

# Chat template used by the original notebook:
#   * adds a `tool` role (for FUNCTION RESPONSE turns in the glaive dataset)
#   * adds {% generation %} tags so `assistant_only_loss=True` in TRL only
#     computes the loss on assistant tokens.
PATCHED_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'tool' %}{{ '<|im_start|>tool\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def load_tokenizer(model_id: str, padding_side: str = "right"):
    """Load the tokenizer, set a pad token and install the patched chat template."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    tokenizer.chat_template = PATCHED_CHAT_TEMPLATE
    return tokenizer
