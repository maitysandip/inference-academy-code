# Inference Academy: code

[![tests](https://github.com/maitysandip/inference-academy-code/actions/workflows/test.yml/badge.svg)](https://github.com/maitysandip/inference-academy-code/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Small, readable implementations of how LLMs are built, served and measured: a Llama-style inference engine with a KV cache in about 120 lines, a GGUF reader using `mmap`, INT8 and 4-bit quantised matmul, continuous batching, an open-loop load generator that reports goodput, a tiny GPT trained with its own autograd, and more.

Almost everything uses the Python standard library only, so you can read every line and nothing is hidden. Each file is explained step by step in a lesson at [Inference Academy](https://inferenceacademy.com).

## Start here

```bash
git clone https://github.com/maitysandip/inference-academy-code && cd inference-academy-code
python3 inference_engine/engine.py          # a tiny Llama-style engine; checks KV cache == recompute
python3 benchmarking/roofline.py            # decode speed ceilings per GPU from bytes and FLOPs
python3 -m unittest discover -s tests -v    # 41 fast tests
```

| If you want to... | Read | Then the lesson |
|---|---|---|
| See what an inference engine does, end to end | [`inference_engine/engine.py`](inference_engine/engine.py) | [The forward pass end to end](https://inferenceacademy.com/learn/engine-core/the-forward-pass-end-to-end) |
| Predict LLM speed before running anything | [`benchmarking/roofline.py`](benchmarking/roofline.py) | [Prefill, decode and the roofline](https://inferenceacademy.com/learn/inf-bottlenecks/prefill-decode-and-the-roofline) |
| Load-test a vLLM, SGLang or llama.cpp server properly | [`benchmarking/loadgen.py`](benchmarking/loadgen.py) | [Benchmarking done properly](https://inferenceacademy.com/learn/inf-benchmarking/benchmarking-done-properly) |
| Write a tokenizer | [`llm_from_scratch/bpe.py`](llm_from_scratch/bpe.py) | [Tokens and byte-pair encoding](https://inferenceacademy.com/learn/llm-text-to-tensors/tokens-and-byte-pair-encoding) |
| Speak MCP over stdio with no SDK | [`llm_apps/mcp_minimal.py`](llm_apps/mcp_minimal.py) | [MCP servers and clients, from the wire up](https://inferenceacademy.com/learn/apps-agents/mcp-servers-and-clients) |

## What is inside

| Folder | What is inside | Course |
|---|---|---|
| `inference_engine/` | Llama-style forward pass with RMSNorm, RoPE, GQA, SwiGLU and a KV cache; a GGUF reader/writer using `mmap`; INT8 and 4-bit quantisation; chat templates; grammar-constrained decoding; batching simulation; a streaming server; a C matvec benchmark | [Build an Inference Engine](https://inferenceacademy.com/courses/build-engine) |
| `benchmarking/` | An open-loop load generator for OpenAI-compatible streaming servers (TTFT, TPOT percentiles, goodput at SLOs), a fake streaming server to check it, closed- versus open-loop statistics, roofline estimates | [Inference Engineering](https://inferenceacademy.com/inference-engineering) |
| `llm_from_scratch/` | BPE tokenizer, causal and online-softmax attention, a complete 4,128-parameter GPT with autograd and Adam (`microgpt.py`), sampling, RoPE and LoRA, MinHash deduplication, fp16 and bf16 | [Build an LLM from Scratch](https://inferenceacademy.com/courses/build-llm) |
| `reasoning/` | pass@k, majority voting, best-of-n with a verifier, a toy GRPO, test-time scaling, process reward models with tree search, a cost-versus-accuracy simulator | [Build a Reasoning Model](https://inferenceacademy.com/courses/build-reasoning) |
| `finetuning/` | LoRA maths, training memory, ZeRO and FSDP sharding, learning-rate and forgetting experiments, a synthetic data pipeline, a tool-call harness, toy contrastive embeddings, and a real LoRA fine-tune of a small model on CPU (`lab/`, needs PyTorch) | [Fine-tuning and Customisation](https://inferenceacademy.com/courses/finetune) |
| `llm_apps/` | A minimal MCP server and client, context packing, SSE streaming, cost and latency budgets, document chunking, a RAG evaluation loop, an agent lab | [Building LLM Applications](https://inferenceacademy.com/courses/llm-apps) |
| `evals/` | Agent trajectory scoring; cost and latency evaluation with percentiles and SLO checks | [LLM Evaluation and Testing](https://inferenceacademy.com/courses/evals) |
| `security/` | Secure RAG with access control, chunk screening and output checks; differential privacy on counts | [AI Security and Privacy](https://inferenceacademy.com/courses/security) |
| `multimodal/` | OCR metrics: character and word error rate, field and table-cell accuracy | [Multimodal Inference](https://inferenceacademy.com/courses/multimodal) |
| `production/` | Serverless versus dedicated GPU cost and cold starts | [Serving AI in Production](https://inferenceacademy.com/courses/production) |

## Where each file is explained

Every file has a lesson that walks through it. Free lessons are readable without an account.

| File | Lesson | Lesson access |
|---|---|---|
| [`benchmarking/bench_stats.py`](benchmarking/bench_stats.py) | [Benchmarking done properly: open loop, percentiles, goodput and honest reporting](https://inferenceacademy.com/learn/inf-benchmarking/benchmarking-done-properly) | Free |
| [`benchmarking/fake_server.py`](benchmarking/fake_server.py) | [Benchmarking done properly: open loop, percentiles, goodput and honest reporting](https://inferenceacademy.com/learn/inf-benchmarking/benchmarking-done-properly) | Free |
| [`benchmarking/loadgen.py`](benchmarking/loadgen.py) | [Capstone lab: from baseline to 3x goodput](https://inferenceacademy.com/learn/inf-benchmarking/lab-baseline-to-3x-throughput) | Pro |
| [`benchmarking/roofline.py`](benchmarking/roofline.py) | [Prefill, decode and the roofline: predicting inference speed](https://inferenceacademy.com/learn/inf-bottlenecks/prefill-decode-and-the-roofline) | Free |
| [`evals/cost_latency_eval.py`](evals/cost_latency_eval.py) | [Cost and latency evaluation](https://inferenceacademy.com/learn/eval-practice/cost-and-latency-evaluation) | Pro |
| [`evals/trajectory_eval.py`](evals/trajectory_eval.py) | [Evaluating agent trajectories](https://inferenceacademy.com/learn/eval-practice/evaluating-agent-trajectories) | Pro |
| [`finetuning/forgetting.py`](finetuning/forgetting.py) | [Forgetting and choosing a method: full fine-tuning, LoRA, replay and weight interpolation](https://inferenceacademy.com/learn/ft-lora/forgetting-and-choosing-a-method) | Pro |
| [`finetuning/lab/`](finetuning/lab/) | [Lab: fine-tune a model with LoRA using PEFT and TRL, from data to a merged model](https://inferenceacademy.com/learn/ft-labs/lab-lora-fine-tune-with-peft-and-trl) | Pro |
| [`finetuning/learning_rate_and_overfitting.py`](finetuning/learning_rate_and_overfitting.py) | [The training recipe: loss masking, packing, learning rate, epochs and overfitting](https://inferenceacademy.com/learn/ft-lora/training-recipe-hyperparameters-and-loss-masking) | Pro |
| [`finetuning/lora_math.py`](finetuning/lora_math.py) | [LoRA and QLoRA in depth: rank, alpha, target layers and 4-bit base models](https://inferenceacademy.com/learn/ft-lora/lora-and-qlora-in-depth) | Free |
| [`finetuning/synthetic_data_pipeline.py`](finetuning/synthetic_data_pipeline.py) | [Lab: build a synthetic data pipeline: validate, deduplicate, decontaminate and balance](https://inferenceacademy.com/learn/ft-labs/lab-synthetic-data-pipeline) | Pro |
| [`finetuning/tool_call_harness.py`](finetuning/tool_call_harness.py) | [Fine-tuning for structured output and tool calling: data, grammars and an evaluation harness](https://inferenceacademy.com/learn/ft-labs/fine-tuning-for-structured-output-and-tool-calling) | Pro |
| [`finetuning/toy_embeddings.py`](finetuning/toy_embeddings.py) | [Fine-tuning embedding models and rerankers: contrastive loss, hard negatives and retrieval metrics](https://inferenceacademy.com/learn/ft-labs/fine-tuning-embeddings-and-rerankers) | Pro |
| [`finetuning/training_memory.py`](finetuning/training_memory.py) | [How much GPU do you need to fine-tune? Memory for full fine-tuning, LoRA and QLoRA](https://inferenceacademy.com/learn/ft-lora/how-much-gpu-to-fine-tune) | Pro |
| [`finetuning/zero_sharding.py`](finetuning/zero_sharding.py) | [Full fine-tuning and sharding across GPUs](https://inferenceacademy.com/learn/ft-lora/full-fine-tuning-and-sharding) | Pro |
| [`inference_engine/batching.py`](inference_engine/batching.py) | [Continuous batching and an OpenAI-style streaming server](https://inferenceacademy.com/learn/engine-fast/batching-and-the-server) | Pro |
| [`inference_engine/chat_template.py`](inference_engine/chat_template.py) | [Tokenizers and chat templates for real models](https://inferenceacademy.com/learn/engine-core/tokenizers-and-chat-templates) | Pro |
| [`inference_engine/constrained_decoding.py`](inference_engine/constrained_decoding.py) | [Grammar-constrained decoding inside an engine](https://inferenceacademy.com/learn/engine-core/grammar-constrained-decoding) | Pro |
| [`inference_engine/engine.py`](inference_engine/engine.py) | [The forward pass end to end: a Llama-style engine in 100 lines](https://inferenceacademy.com/learn/engine-core/the-forward-pass-end-to-end) | Free |
| [`inference_engine/gguf_mini.py`](inference_engine/gguf_mini.py) | [Loading weights: the GGUF file format and memory mapping](https://inferenceacademy.com/learn/engine-core/loading-weights-gguf-and-mmap) | Pro |
| [`inference_engine/matvec_bench.c`](inference_engine/matvec_bench.c) | [Speed on the CPU: SIMD, threads, cache layout and profiling](https://inferenceacademy.com/learn/engine-fast/speed-threads-simd-and-cache-layout) | Pro |
| [`inference_engine/quant.py`](inference_engine/quant.py) | [Quantised matrix multiplication: INT8 and 4-bit kernels](https://inferenceacademy.com/learn/engine-fast/quantised-matmul-int8-and-int4) | Pro |
| [`inference_engine/server.py`](inference_engine/server.py) | [Continuous batching and an OpenAI-style streaming server](https://inferenceacademy.com/learn/engine-fast/batching-and-the-server) | Pro |
| [`llm_apps/agent_lab.py`](llm_apps/agent_lab.py) | [Lab: an agent with MCP tools, guards and trajectory checks](https://inferenceacademy.com/learn/apps-shipping/lab-agent-with-mcp-tools-and-evals) | Pro |
| [`llm_apps/budgets.py`](llm_apps/budgets.py) | [Cost and latency budgets for LLM features](https://inferenceacademy.com/learn/apps-shipping/cost-and-latency-budgets) | Pro |
| [`llm_apps/context_packing.py`](llm_apps/context_packing.py) | [Context engineering: deciding what the model sees](https://inferenceacademy.com/learn/apps-agents/context-engineering) | Pro |
| [`llm_apps/doc_chunking.py`](llm_apps/doc_chunking.py) | [Document processing for RAG: parsing, OCR and structure-aware chunks](https://inferenceacademy.com/learn/apps-shipping/document-processing-for-rag) | Pro |
| [`llm_apps/mcp_minimal.py`](llm_apps/mcp_minimal.py) | [MCP servers and clients, from the wire up](https://inferenceacademy.com/learn/apps-agents/mcp-servers-and-clients) | Free |
| [`llm_apps/rag_eval.py`](llm_apps/rag_eval.py) | [The RAG evaluation loop](https://inferenceacademy.com/learn/apps-shipping/rag-evaluation-loop) | Pro |
| [`llm_apps/streaming_sse.py`](llm_apps/streaming_sse.py) | [Streaming UX: making slow models feel fast](https://inferenceacademy.com/learn/apps-shipping/streaming-ux) | Free |
| [`llm_from_scratch/attention.py`](llm_from_scratch/attention.py) | [Attention from first principles: queries, keys, values and the causal mask](https://inferenceacademy.com/learn/llm-attention-and-gpt/attention-from-first-principles) | Pro |
| [`llm_from_scratch/bpe.py`](llm_from_scratch/bpe.py) | [Tokens and byte-pair encoding: write a tokenizer](https://inferenceacademy.com/learn/llm-text-to-tensors/tokens-and-byte-pair-encoding) | Free |
| [`llm_from_scratch/dedup.py`](llm_from_scratch/dedup.py) | [A real data pipeline: cleaning, filtering and deduplication](https://inferenceacademy.com/learn/llm-modern/a-real-data-pipeline-cleaning-and-dedup) | Pro |
| [`llm_from_scratch/microgpt.py`](llm_from_scratch/microgpt.py) | [Lab: train a tiny GPT in pure Python](https://inferenceacademy.com/learn/llm-training/lab-train-a-tiny-gpt) | Pro |
| [`llm_from_scratch/precision.py`](llm_from_scratch/precision.py) | [Mixed precision and multi-GPU training basics](https://inferenceacademy.com/learn/llm-modern/mixed-precision-and-multi-gpu-basics) | Pro |
| [`llm_from_scratch/rope_lora.py`](llm_from_scratch/rope_lora.py) | [Modern upgrades: RoPE, RMSNorm, SwiGLU and grouped-query attention](https://inferenceacademy.com/learn/llm-modern/modern-upgrades-rope-rmsnorm-swiglu-gqa) | Pro |
| [`llm_from_scratch/sampling.py`](llm_from_scratch/sampling.py) | [Generation: sampling strategies and the KV cache](https://inferenceacademy.com/learn/llm-training/generation-sampling-and-kv-cache) | Pro |
| [`multimodal/ocr_metrics.py`](multimodal/ocr_metrics.py) | [Document AI: OCR, layout, tables and extraction](https://inferenceacademy.com/learn/mm-vision-language/document-ai-ocr-and-layout) | Pro |
| [`production/serverless_vs_dedicated.py`](production/serverless_vs_dedicated.py) | [Serverless GPU platforms: when they save money and when they cost more](https://inferenceacademy.com/learn/prod-architecture/serverless-gpu-platforms) | Pro |
| [`reasoning/cost_vs_accuracy.py`](reasoning/cost_vs_accuracy.py) | [Lab: cost versus accuracy for reasoning strategies](https://inferenceacademy.com/learn/reason-training/lab-cost-versus-accuracy) | Pro |
| [`reasoning/methods.py`](reasoning/methods.py) | [Chain of thought and self-consistency: majority voting, and when it fails](https://inferenceacademy.com/learn/reason-inference/chain-of-thought-and-self-consistency) | Pro |
| [`reasoning/prm_search.py`](reasoning/prm_search.py) | [Process reward models and tree search](https://inferenceacademy.com/learn/reason-inference/process-reward-models-and-tree-search) | Pro |
| [`reasoning/test_time_scaling.py`](reasoning/test_time_scaling.py) | [Test-time scaling: spending compute where it helps](https://inferenceacademy.com/learn/reason-inference/test-time-scaling) | Pro |
| [`security/dp_counts.py`](security/dp_counts.py) | [Privacy technology and AI governance](https://inferenceacademy.com/learn/sec-defense/privacy-technology-and-governance) | Pro |
| [`security/secure_rag.py`](security/secure_rag.py) | [Guardrails in practice and secure RAG](https://inferenceacademy.com/learn/sec-defense/guardrails-and-secure-rag) | Pro |

## Run

```bash
python3 -m unittest discover -s tests -v        # 41 fast tests
python3 llm_from_scratch/microgpt.py            # trains the tiny GPT (about 2 minutes, pure Python)
python3 inference_engine/engine.py              # KV cache equals recompute check
python3 reasoning/cost_vs_accuracy.py           # strategy comparison table
python3 benchmarking/bench_stats.py             # closed vs open loop percentiles
python3 benchmarking/roofline.py                # decode ceilings per GPU
cc -O3 -march=native -ffast-math -o matvec inference_engine/matvec_bench.c && ./matvec
```

Load-test a real server (any OpenAI-compatible streaming endpoint):

```bash
python3 benchmarking/loadgen.py --url http://localhost:8000/v1/chat/completions --model YOUR_MODEL \
        --rates 1,2,4,8 --duration 120 --slo-ttft 1.0 --slo-tpot 0.06
```

It prints one JSON line per request rate: TTFT p50/p95/p99, TPOT p95, tokens per second and goodput (requests per second that met both SLOs). Arrivals are Poisson and latency is measured from the scheduled arrival time, so a slow server cannot hide its queueing delay.

## Honest notes

- These are teaching implementations. They are far too slow for real models and the engine uses random weights; the tests check mechanisms (cache equals recompute, causal masking, merge identities), not text quality. `loadgen.py` is the exception: it is meant for real servers.
- The GGUF code implements a small subset of the format (strings, `uint32`, float32 tensors). Read the specification before relying on it for real files.
- GPU specifications in `roofline.py` are rounded datasheet values; check current numbers.
- Benchmark numbers in the lessons come from the author's laptop and will differ on yours.

## Contributing

Bug reports and fixes are welcome: open an issue or a pull request. Keep files dependency-free and small enough to read in one sitting, and add a test for any behaviour you change.

## Licence

MIT, see [`LICENSE`](LICENSE). If you use this code in teaching material or a blog post, a link back to [inferenceacademy.com](https://inferenceacademy.com) is appreciated.
