"""Chat templates: how a list of messages becomes the exact token sequence a model was trained on.

Instruction-tuned models were trained on conversations rendered in one specific format. Feed them a
different format and quality drops quietly. This script renders the same conversation in two common
template styles, shows the generation prompt, and demonstrates two classic bugs:
  1. a double BOS token (template adds one, tokenizer adds another)
  2. a missing generation prompt, so the model continues the user's turn instead of answering
Real templates are Jinja programs shipped in tokenizer_config.json or the GGUF metadata; the logic is the same.
"""

MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Italy?"},
]


def chatml(messages, add_generation_prompt=True):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def llama3(messages, add_generation_prompt=True, bos=True):
    s = "<|begin_of_text|>" if bos else ""
    for m in messages:
        s += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>"
    return s + ("<|start_header_id|>assistant<|end_header_id|>\n\n" if add_generation_prompt else "")


def tokenize(text, add_bos):
    """A stand-in tokenizer: special tokens are single ids, everything else is split on spaces.
    Like real tokenizers, it can prepend a BOS token by default."""
    import re
    parts = re.findall(r"<\|[a-z_]+\|>|\S+|\n", text)
    return (["<|begin_of_text|>"] if add_bos else []) + parts


if __name__ == "__main__":
    print("== ChatML style ==")
    print(chatml(MESSAGES))
    print("== Llama 3 style (line breaks after <|eot_id|> added for display) ==")
    print(llama3(MESSAGES).replace("<|eot_id|>", "<|eot_id|>\n"))

    print("\n== bug 1: double BOS ==")
    ids = tokenize(llama3(MESSAGES), add_bos=True)
    print("first tokens:", ids[:3], "-> BOS count:", ids.count("<|begin_of_text|>"))
    ids = tokenize(llama3(MESSAGES), add_bos=False)
    print("fixed (add_special_tokens=False when the template already has BOS):", ids[:2], "-> BOS count:", ids.count("<|begin_of_text|>"))

    print("\n== bug 2: no generation prompt ==")
    tail = chatml(MESSAGES, add_generation_prompt=False)[-40:]
    print("prompt ends with:", repr(tail))
    print("the model's most likely continuation is another turn header or more user text, not an answer")

    print("\n== stop tokens differ per template ==")
    print("ChatML stops on <|im_end|>; Llama 3 stops on <|eot_id|>. A server that only knows the tokenizer's EOS keeps generating.")
