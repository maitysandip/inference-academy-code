# Inference Academy: code

Small, dependency-free Python implementations that accompany the courses at Inference Academy. Everything runs with the standard library only (Python 3.9+), so you can read every line and nothing is hidden.

| Folder | What is inside | Related course |
|---|---|---|
| `llm_from_scratch/` | BPE tokenizer, causal and online-softmax attention, a complete 4,128-parameter GPT with autograd and Adam (`microgpt.py`), sampling, RoPE and LoRA | Build an LLM from Scratch |
| `inference_engine/` | Llama-style forward pass with GQA and a KV cache, a GGUF reader/writer using `mmap`, INT8 and 4-bit quantisation, batching simulation, a streaming server, a C matvec benchmark | Build an Inference Engine |
| `reasoning/` | pass@k, majority voting, best-of-n with a verifier, a toy GRPO, a cost-versus-accuracy simulator | Build a Reasoning Model |
| `benchmarking/` | An open-loop load generator, a fake streaming server, latency statistics, closed-loop versus open-loop demonstration, roofline estimates | Inference Engineering |
| `finetuning/` | LoRA maths, training memory, ZeRO and FSDP sharding, learning-rate and forgetting experiments, a synthetic data pipeline, a tool-call harness, toy contrastive embeddings, and a real LoRA fine-tune of a small model on CPU (needs PyTorch) | Fine-tuning and Customisation |
| `llm_apps/` | A minimal MCP server and client, context packing, SSE streaming, cost and latency budgets, document chunking, a RAG evaluation loop, an agent lab | Building LLM Applications |
| `security/` | Secure RAG with access control, chunk screening and output checks; differential privacy on counts | AI Security and Privacy |
| `evals/` | Agent trajectory scoring; cost and latency evaluation with percentiles and SLO checks | LLM Evaluation and Testing |
| `multimodal/` | OCR metrics: character and word error rate, field and table-cell accuracy | Multimodal Inference |
| `production/` | Serverless versus dedicated GPU cost and cold starts | Serving AI in Production |

The Build courses also gained `llm_from_scratch/dedup.py` (MinHash deduplication) and `precision.py` (fp16 and bf16), `inference_engine/chat_template.py` and `constrained_decoding.py`, and `reasoning/test_time_scaling.py` and `prm_search.py`.

## Run

```bash
python3 -m unittest discover -s tests -v        # 16 fast tests
python3 llm_from_scratch/microgpt.py            # trains the tiny GPT (about 2 minutes, pure Python)
python3 inference_engine/engine.py              # KV cache equals recompute check
python3 reasoning/cost_vs_accuracy.py           # strategy comparison table
python3 benchmarking/bench_stats.py             # closed vs open loop percentiles
python3 benchmarking/roofline.py                # decode ceilings per GPU
cc -O3 -march=native -ffast-math -o matvec inference_engine/matvec_bench.c && ./matvec
```

## Honest notes

- These are teaching implementations. They are far too slow for real models and the engine uses random weights; the tests check mechanisms (cache equals recompute, causal masking, merge identities), not text quality.
- The GGUF code implements a small subset of the format (strings, `uint32`, float32 tensors). Read the specification before relying on it for real files.
- GPU specifications in `roofline.py` are rounded datasheet values; check current numbers.
- Benchmark numbers in the lessons come from the author's laptop and will differ on yours.

## Licence

MIT, see `LICENSE`.
