import json, time, torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM = 'Classify the support ticket. Reply with JSON only, like {"category": "billing", "priority": "low"}.'
torch.manual_seed(0)
torch.set_num_threads(8)

data = json.load(open("tickets.json"))
tok = AutoTokenizer.from_pretrained(MODEL)

def prompt(ticket):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ticket}]

def to_example(r):
    return {"prompt": prompt(r["ticket"]), "completion": [{"role": "assistant", "content": json.dumps(r["label"])}]}

train = Dataset.from_list([to_example(r) for r in data["train"]])

@torch.no_grad()
def evaluate(model, rows, label):
    model.eval()
    ok = dict(valid=0, category=0, priority=0, exact=0)
    for r in rows:
        ids = tok.apply_chat_template(prompt(r["ticket"]), add_generation_prompt=True, return_tensors="pt", return_dict=True)
        out = model.generate(**ids, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            got = json.loads(text); ok["valid"] += 1
            c, p = got.get("category") == r["label"]["category"], got.get("priority") == r["label"]["priority"]
            ok["category"] += c; ok["priority"] += p; ok["exact"] += c and p
        except Exception:
            pass
    n = len(rows)
    print(f"{label:<14} valid JSON {ok['valid']/n:5.0%}  category {ok['category']/n:5.0%}  priority {ok['priority']/n:5.0%}  exact {ok['exact']/n:5.0%}")
    return {k: v / n for k, v in ok.items()}

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
before = evaluate(base, data["test"], "base model")

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")
args = SFTConfig(output_dir="out", num_train_epochs=2, per_device_train_batch_size=8, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=5,
                 logging_steps=10, save_strategy="no", report_to="none", max_length=256, seed=0, use_cpu=True, bf16=False, gradient_checkpointing=False)
trainer = SFTTrainer(model=base, args=args, train_dataset=train, peft_config=lora, processing_class=tok)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
print(f"trainable parameters: {trainable:,} of {total:,} ({trainable/total:.2%})")
t0 = time.time(); trainer.train(); print(f"training took {time.time()-t0:.0f}s")

after = evaluate(trainer.model, data["test"], "LoRA tuned")
trainer.model.save_pretrained("adapter")
print("adapter files:", sorted(__import__("os").listdir("adapter")))
import os; print("adapter size: %.1f MB" % (sum(os.path.getsize(f"adapter/{f}") for f in os.listdir("adapter")) / 1e6))

# Reload the adapter on a fresh base model and merge it into one standalone model.
fresh = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
merged = PeftModel.from_pretrained(fresh, "adapter").merge_and_unload()
evaluate(merged, data["test"], "merged model")
merged.save_pretrained("merged"); tok.save_pretrained("merged")
