"""Run the base and the tuned (merged) model on the test tickets and on a small regression suite; save every output."""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM = 'Classify the support ticket. Reply with JSON only, like {"category": "billing", "priority": "low"}.'
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MODEL)
models = {"base": AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32), "tuned": AutoModelForCausalLM.from_pretrained("merged", dtype=torch.float32)}
for m in models.values(): m.eval()

@torch.no_grad()
def reply(model, messages, max_new_tokens):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

test = json.load(open("tickets.json"))["test"]
REGRESSION = [  # (prompt, words that a correct answer should contain)
    ("What is the capital of France?", ["paris"]), ("What is 17 + 25?", ["42"]), ("What does CPU stand for?", ["central processing unit"]),
    ("Name three fruits.", ["apple", "banana", "orange", "mango", "grape", "pear"]), ("Translate 'good morning' into Spanish.", ["buenos d"]),
    ("Name a primary colour.", ["red", "blue", "yellow"]), ("Is the Earth round?", ["round", "sphere", "spherical", "oblate", "yes"]),
    ("Write a one-line Python function that adds two numbers.", ["def", "lambda"]), ("Who wrote the play Romeo and Juliet?", ["shakespeare"]),
    ("What is the opposite of hot?", ["cold"]), ("How many days are in a week?", ["7", "seven"]), ("Write a short polite sentence declining a meeting.", ["sorry", "unable", "regret", "decline", "cannot", "can't", "unfortunately", "thank"]),
]
out = {"test": [], "regression": []}
for r in test:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["ticket"]}]
    out["test"].append({"ticket": r["ticket"], "label": r["label"], "base": reply(models["base"], msgs, 40), "tuned": reply(models["tuned"], msgs, 40)})
for q, kw in REGRESSION:
    msgs = [{"role": "user", "content": q}]
    out["regression"].append({"prompt": q, "keywords": kw, "base": reply(models["base"], msgs, 60), "tuned": reply(models["tuned"], msgs, 60)})
json.dump(out, open("predictions.json", "w"), indent=1)
print("saved", len(out["test"]), "test and", len(out["regression"]), "regression outputs")
