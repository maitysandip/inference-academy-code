# Fine-tuning code

Companion code for the Fine-tuning and Customisation course. Everything in this folder was run to produce the numbers printed in the lessons.

## Pure Python (standard library only)

| Script | Lesson |
|---|---|
| `lora_math.py` | LoRA parameter counts, adapter equals merged weights, low-rank approximation, 4-bit grids |
| `training_memory.py` | Memory for full fine-tuning, LoRA and QLoRA |
| `learning_rate_and_overfitting.py` | Learning-rate sweep, epochs, weight decay |
| `forgetting.py` | Catastrophic forgetting, replay, anchoring, weight interpolation |
| `synthetic_data_pipeline.py` | The synthetic data pipeline lab |
| `tool_call_harness.py` | Validator and metrics for tool calling |
| `toy_embeddings.py [seed]` | Contrastive training with hard negatives |

```bash
python3 finetuning/lora_math.py
python3 finetuning/synthetic_data_pipeline.py
```

## The real LoRA lab (needs PyTorch)

Tested with Python 3.12, torch 2.15, transformers 5.23, peft 0.21, trl 1.18, datasets 5.2. It runs on a laptop CPU in about three minutes.

```bash
pip install torch transformers peft trl datasets accelerate
cd finetuning/lab
python make_data.py      # builds tickets.json
python train_lora.py     # baseline, LoRA training, evaluation, adapter, merged model (about 3 minutes on CPU)
python predict.py        # saves predictions.json for base and merged models
python evaluate.py       # intervals, paired test, error analysis, regression suite
python token_stats.py    # prompt versus answer tokens and padding waste
```

Library APIs change quickly. If a call fails, check the release notes for the version you installed. On a CPU keep `bf16=False` and `gradient_checkpointing=False` (the TRL defaults are for GPUs).
