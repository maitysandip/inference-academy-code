# Inference Academy: code

Small, dependency-free Python implementations that accompany the courses at Inference Academy. Everything runs with the standard library only (Python 3.9+), so you can read every line and nothing is hidden.

| Folder | What is inside | Related course |
|---|---|---|
| `llm_from_scratch/` | BPE tokenizer, causal and online-softmax attention, a complete 4,128-parameter GPT with autograd and Adam (`microgpt.py`), sampling, RoPE and LoRA | Build an LLM from Scratch |
| `inference_engine/` | Llama-style forward pass with GQA and a KV cache, a GGUF reader/writer using `mmap`, INT8 and 4-bit quantisation, batching simulation, a streaming server, a C matvec benchmark | Build an Inference Engine |
| `reasoning/` | pass@k, majority voting, best-of-n with a verifier, a toy GRPO, a cost-versus-accuracy simulator | Build a Reasoning Model |
| `benchmarking/` | An open-loop load generator, a fake streaming server, latency statistics, closed-loop versus open-loop demonstration, roofline estimates | Inference Engineering |

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
