"""A small simulator of one LLM serving replica, for incident practice. Standard library only.

It models the parts of a vLLM-style engine that cause most latency and throughput incidents:
a paged KV cache with a fixed pool of blocks, prefix caching, a scheduler with a token budget
and a sequence limit, chunked prefill, and preemption with recompute when the pool runs out.
Step time comes from a simple roofline: read the weights and the KV cache once per step, or
do the prefill maths, whichever takes longer, plus a fixed CPU and launch overhead.

It is a teaching model, not a replica of vLLM. The numbers are what the formulas below give
for one 48 GB GPU, not measurements. Use it to practise the method, then check on real systems.

Usage:  python3 incident_sim.py SCENARIO COMMAND [--set key=value ...] [--seed N]
        python3 incident_sim.py list
SCENARIO is one of the names in scenarios.py; COMMAND is brief, dashboard, changes, config,
startup-log, metrics, requests, sweep or reveal.
"""
import heapq, math, random, sys
from dataclasses import dataclass, field, fields, replace

GB = 1e9


@dataclass
class Engine:
    """The replica: hardware, model and engine settings (named after vLLM's engine arguments)."""
    gpu_memory_gb: float = 48.0            # total memory on the card
    hbm_gbps: float = 864.0                # memory bandwidth, GB/s
    tflops: float = 362.0                  # dense FP8 tensor throughput, TFLOP/s
    bw_efficiency: float = 0.80            # share of bandwidth a decode step really gets
    mfu: float = 0.40                      # share of peak FLOPs a prefill really gets
    params_b: float = 34.0                 # parameters, billions
    weights_gb: float = 35.0               # weights in memory (FP8 plus embeddings)
    layers: int = 64
    kv_heads: int = 8
    head_dim: int = 128
    activation_gb: float = 1.6             # peak activation memory found by the startup profile
    non_torch_gb: float = 0.5              # CUDA context, NCCL buffers and the like
    cuda_graph_gb: float = 0.6             # memory held by captured CUDA graphs
    gpu_memory_utilization: float = 0.90
    kv_cache_dtype: str = "auto"           # "auto" stores KV in BF16 (2 bytes); "fp8" uses 1 byte
    block_size: int = 16
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    max_model_len: int = 8192
    enable_prefix_caching: bool = True
    enforce_eager: bool = False            # True disables CUDA graphs
    step_overhead_ms: float = 3.0          # scheduler and Python work per step
    eager_overhead_ms: float = 8.0         # extra kernel-launch time per step without CUDA graphs

    def kv_bytes_per_token(self):
        return 2 * self.layers * self.kv_heads * self.head_dim * (1 if self.kv_cache_dtype == "fp8" else 2)

    def memory_plan(self):
        """How the engine splits memory at startup: everything left in the budget becomes KV cache."""
        budget = self.gpu_memory_utilization * self.gpu_memory_gb
        graphs = 0.0 if self.enforce_eager else self.cuda_graph_gb
        kv_gb = budget - self.weights_gb - self.activation_gb - self.non_torch_gb - graphs
        blocks = max(0, int(kv_gb * GB / (self.kv_bytes_per_token() * self.block_size)))
        return {"budget_gb": budget, "graphs_gb": graphs, "kv_gb": kv_gb, "blocks": blocks, "kv_tokens": blocks * self.block_size}

    def step_seconds(self, kv_tokens_read, tokens_computed):
        memory = (self.weights_gb * GB + kv_tokens_read * self.kv_bytes_per_token()) / (self.hbm_gbps * GB * self.bw_efficiency)
        compute = 2 * self.params_b * 1e9 * tokens_computed / (self.tflops * 1e12 * self.mfu)
        overhead = self.step_overhead_ms + (self.eager_overhead_ms if self.enforce_eager else 0.0)
        return overhead / 1000 + max(memory, compute)


@dataclass
class Workload:
    """The traffic. Lengths are lognormal; a shared system prompt can sit in front of each request."""
    rate: float = 1.0                      # requests per second (open loop)
    clients: int = 0                       # if above 0: closed loop with this many clients, rate ignored
    duration_s: float = 900.0
    warmup_s: float = 120.0                # results ignore requests that arrive before this
    prompt_mean: float = 450.0             # tokens of user text, after any system prompt
    prompt_sigma: float = 0.6
    output_mean: float = 250.0
    output_sigma: float = 0.6
    max_tokens: int = 1024
    system_prompts: int = 3                # distinct shared system prompts (0 = none)
    system_prompt_len: int = 1536
    header_tokens: int = 24                # per-request metadata (time, request id) in the template
    header_position: str = "end"           # "end" keeps the shared prefix intact; "start" puts it first
    timeout_s: float = 0.0                 # client gives up (and the engine aborts the request) after this; 0 = never
    gateway_limit: int = 0                 # admission control: reject at once beyond this many requests in the engine; 0 = off


@dataclass
class Request:
    rid: int
    arrival: float
    prompt_len: int                        # total prompt tokens, system prompt included
    out_len: int
    prefix_key: object                     # cache key of the shared prefix, or None
    prefix_len: int
    client: int = -1
    computed: int = 0                      # tokens whose KV is in the cache
    generated: int = 0
    own_blocks: int = 0                    # blocks this request holds privately
    holds_prefix: bool = False             # holds a reference to the cached shared prefix
    own_from_zero: bool = False            # its cached private blocks start at token 0
    hit_tokens: int = 0
    queried_tokens: int = 0
    preemptions: int = 0
    first_token: float = None
    last_token: float = None
    finish: float = None
    gaps: list = field(default_factory=list)

    def prefill_target(self):              # tokens to compute before the next token can be produced
        return self.prompt_len + max(0, self.generated - 1)


class BlockPool:
    """A fixed pool of KV blocks plus a prefix cache of blocks kept after use, evicted least recently used first."""
    def __init__(self, blocks):
        self.total = blocks
        self.free = blocks
        self.cache = {}                    # key -> [blocks, refcount, last_used]
        self.clock = 0

    def evictable(self, keep=()):
        return sum(b for k, (b, rc, _) in self.cache.items() if rc == 0 and k not in keep)

    def take(self, n, keep=()):
        if self.free + self.evictable(keep) < n:
            return False
        while self.free < n:
            k = min((k for k, (b, rc, _) in self.cache.items() if rc == 0 and k not in keep), key=lambda k: self.cache[k][2])
            self.free += self.cache.pop(k)[0]
        self.free -= n
        return True

    def touch(self, key):
        self.clock += 1
        self.cache[key][2] = self.clock

    def used(self):                        # blocks in use by running requests (cached, unreferenced blocks count as free)
        return self.total - self.free - self.evictable()


class Replica:
    def __init__(self, engine: Engine, workload: Workload, seed=1):
        self.e, self.w = engine, workload
        self.rng = random.Random(seed)
        plan = engine.memory_plan()
        if plan["blocks"] * engine.block_size < engine.max_model_len:
            raise SystemExit(f"Startup failed: the KV cache holds {plan['kv_tokens']} tokens, fewer than max_model_len={engine.max_model_len}. "
                             "Lower max_model_len or raise gpu_memory_utilization.")
        self.pool = BlockPool(plan["blocks"])
        self.now = 0.0
        self.waiting, self.running, self.done = [], [], []
        self.arrivals = []                 # heap of (time, rid, request)
        self.next_id = 0
        self.stats = {"steps": 0, "busy_s": 0.0, "preemptions": 0, "running_area": 0.0, "waiting_area": 0.0,
                      "kv_area": 0.0, "batch_tokens_area": 0.0, "prompt_tokens_computed": 0, "gen_tokens": 0,
                      "hit_tokens": 0, "queried_tokens": 0, "gen_tokens_window": 0, "timeouts": 0, "rejected": 0, "too_long": 0, "wasted_tokens": 0, "t0": None}

    # ---- traffic ----
    def lognormal(self, mean, sigma):
        return self.rng.lognormvariate(math.log(mean) - sigma * sigma / 2, sigma)

    def new_request(self, t, client=-1):
        w = self.w
        user = max(8, int(self.lognormal(w.prompt_mean, w.prompt_sigma)))
        out = max(1, min(w.max_tokens, int(self.lognormal(w.output_mean, w.output_sigma))))
        sp = w.system_prompt_len if w.system_prompts else 0
        key = None
        if sp and self.e.enable_prefix_caching:
            # A per-request header at the start makes every prompt different from its first token.
            key = ("hdr", self.next_id) if w.header_position == "start" else ("sys", self.rng.randrange(w.system_prompts))
        prefix_len = sp + (w.header_tokens if w.header_position == "start" else 0)
        r = Request(self.next_id, t, prefix_len + user + (w.header_tokens if w.header_position != "start" else 0), out, key, prefix_len, client)
        self.next_id += 1
        heapq.heappush(self.arrivals, (t, r.rid, r))

    def seed_traffic(self):
        w = self.w
        if w.clients:
            for c in range(w.clients):
                self.new_request(self.rng.random() * 0.5, c)
        else:
            t = 0.0
            while True:
                t += self.rng.expovariate(w.rate)
                if t >= w.duration_s:
                    break
                self.new_request(t)

    # ---- KV accounting ----
    def blocks_held(self, r):
        return r.own_blocks + (r.prefix_len // self.e.block_size if r.holds_prefix else 0)

    def need_blocks(self, r, n):
        return max(0, math.ceil((r.computed + n) / self.e.block_size) - self.blocks_held(r))

    def cached_hits(self, r):
        """Blocks this request could reuse now: its shared prefix and, after a preemption, its own old blocks."""
        bs, own = self.e.block_size, ("req", r.rid)
        keys, hit = [], 0
        if own in self.pool.cache and r.own_from_zero:
            keys, hit = [own], self.pool.cache[own][0]
        elif r.prefix_key is not None and r.prefix_key in self.pool.cache:
            keys, hit = [r.prefix_key], r.prefix_len // bs
            if own in self.pool.cache:
                keys.append(own); hit += self.pool.cache[own][0]
        if hit * bs >= r.prefill_target():       # always compute at least one token
            return [], 0
        return keys, hit

    def admit_hits(self, r, keys):
        bs = self.e.block_size
        for k in keys:
            if k == r.prefix_key:
                self.pool.cache[k][1] += 1; self.pool.touch(k); r.holds_prefix = True
                r.computed += r.prefix_len // bs * bs
            else:
                b = self.pool.cache.pop(k)[0]; r.own_blocks += b; r.computed += b * bs

    def release(self, r, keep_for_recompute=False):
        """Free a request's blocks. On preemption its full private blocks stay in the cache, evictable, in case it comes back soon."""
        bs = self.e.block_size
        full = r.computed // bs - (r.prefix_len // bs if r.holds_prefix else 0)
        r.own_from_zero = not r.holds_prefix
        if r.holds_prefix:
            self.pool.cache[r.prefix_key][1] -= 1; r.holds_prefix = False
        if keep_for_recompute and self.e.enable_prefix_caching and 0 < full <= r.own_blocks:
            self.pool.cache[("req", r.rid)] = [full, 0, 0]; self.pool.touch(("req", r.rid))
            self.pool.free += r.own_blocks - full
        else:
            self.pool.free += r.own_blocks
        r.own_blocks = 0

    def preempt(self, r):
        self.running.remove(r)
        self.release(r, keep_for_recompute=True)
        r.computed = 0; r.preemptions += 1
        self.stats["preemptions"] += 1
        self.waiting.insert(0, r)

    def promote_prefix(self, r):
        """Once a request has computed its whole shared prefix, those blocks join the prefix cache."""
        bs = self.e.block_size
        if r.prefix_key is None or r.holds_prefix or r.computed < r.prefix_len:
            return
        nb = r.prefix_len // bs
        if r.prefix_key in self.pool.cache:
            return
        if nb and r.own_blocks >= nb:
            r.own_blocks -= nb
            self.pool.cache[r.prefix_key] = [nb, 1, 0]; self.pool.touch(r.prefix_key); r.holds_prefix = True

    # ---- one engine step ----
    def step(self):
        e, budget, sched, preempted = self.e, self.e.max_num_batched_tokens, [], False
        for r in list(self.running):
            if r not in self.running or budget <= 0:
                continue
            n = min(r.prefill_target() - r.computed, budget) if r.computed < r.prefill_target() else 1
            while not self.pool.take(self.need_blocks(r, n)):
                victim = self.running[-1]  # the most recently admitted request goes first
                self.preempt(victim); preempted = True
                if victim is r:
                    break
            if r not in self.running:
                continue
            r.own_blocks += self.need_blocks(r, n)
            sched.append((r, n)); budget -= n
        while not preempted and self.waiting and len(self.running) < e.max_num_seqs and budget > 0:
            r = self.waiting[0]
            keys, hit = self.cached_hits(r) if e.enable_prefix_caching else ([], 0)
            n = min(r.prefill_target() - hit * e.block_size, budget)
            need = max(0, math.ceil((hit * e.block_size + n) / e.block_size) - hit)
            if not self.pool.take(need, keep=keys):
                break
            self.waiting.pop(0)
            self.admit_hits(r, keys)
            r.own_blocks += need
            q = r.prefill_target()
            r.queried_tokens += q; r.hit_tokens += r.computed
            self.stats["queried_tokens"] += q; self.stats["hit_tokens"] += r.computed
            self.running.append(r); sched.append((r, n)); budget -= n
        if not sched:
            return False
        kv_read = sum(r.computed + n for r, n in sched)
        computed = sum(n for _, n in sched)
        dt = e.step_seconds(kv_read, computed)
        self.account(dt, computed)
        self.now += dt
        for r, n in sched:
            prefill = r.computed < r.prefill_target()
            r.computed += n
            if prefill:
                self.stats["prompt_tokens_computed"] += n
                self.promote_prefix(r)
                if r.computed < r.prefill_target():
                    continue
            self.emit(r)
        return True

    def emit(self, r):
        r.generated += 1; self.stats["gen_tokens"] += 1
        if self.now >= self.w.warmup_s:
            self.stats["gen_tokens_window"] += 1
        if r.first_token is None:
            r.first_token = self.now
        elif r.last_token is not None:
            r.gaps.append(self.now - r.last_token)
        r.last_token = self.now
        if r.generated >= r.out_len:
            r.finish = self.now
            self.running.remove(r); self.release(r); self.done.append(r)
            if self.w.clients and self.now < self.w.duration_s:
                self.new_request(self.now, r.client)

    def expire(self):
        """Clients that waited longer than their timeout disconnect; the engine aborts them and they retry at once."""
        for r in [q for q in self.running + self.waiting if self.now - q.arrival > self.w.timeout_s]:
            if r in self.running:
                self.running.remove(r)
            else:
                self.waiting.remove(r)
            self.release(r)
            if self.now >= self.w.warmup_s:
                self.stats["timeouts"] += 1; self.stats["wasted_tokens"] += r.generated
            if self.w.clients and self.now < self.w.duration_s:
                self.new_request(self.now, r.client)

    def account(self, dt, tokens):
        s = self.stats
        if self.now >= self.w.warmup_s:
            if s["t0"] is None:
                s["t0"] = self.now
            s["busy_s"] += dt; s["steps"] += 1
            s["running_area"] += len(self.running) * dt; s["waiting_area"] += len(self.waiting) * dt
            s["kv_area"] += self.pool.used() / max(1, self.pool.total) * dt; s["batch_tokens_area"] += tokens * dt

    def run(self):
        self.seed_traffic()
        while self.now < self.w.duration_s:
            while self.arrivals and self.arrivals[0][0] <= self.now:
                r = heapq.heappop(self.arrivals)[2]
                if r.prompt_len >= self.e.max_model_len:                    # the engine refuses it: HTTP 400
                    self.stats["too_long"] += self.now >= self.w.warmup_s
                    if self.w.clients:
                        self.new_request(self.now + 1.0, r.client)
                    continue
                r.out_len = min(r.out_len, self.e.max_model_len - r.prompt_len)
                if self.w.gateway_limit and len(self.running) + len(self.waiting) >= self.w.gateway_limit:
                    self.stats["rejected"] += self.now >= self.w.warmup_s   # a fast 429; the client backs off 1 s
                    if self.w.clients:
                        self.new_request(self.now + 1.0, r.client)
                    continue
                self.waiting.append(r)
            if self.w.timeout_s:
                self.expire()
            if not self.step():
                if not self.arrivals:
                    break
                self.now = max(self.now, self.arrivals[0][0])
        return self.report()

    def report(self):
        w, s = self.w, self.stats
        done = [r for r in self.done if r.arrival >= w.warmup_s]
        span = max(1e-9, self.now - (s["t0"] or w.warmup_s))
        unfinished = [r for r in self.running + self.waiting if r.arrival >= w.warmup_s]
        ttft = sorted([r.first_token - r.arrival for r in done] + [self.now - r.arrival for r in unfinished if r.first_token is None])
        gaps = sorted(g for r in done for g in r.gaps)
        e2e = sorted(r.finish - r.arrival for r in done)
        pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
        plan = self.e.memory_plan()
        return {
            "completed": len(done), "unfinished": len(unfinished), "rate_in": (len(done) + len(unfinished)) / span,
            "ttft_p50": pct(ttft, .5), "ttft_p90": pct(ttft, .9), "ttft_p99": pct(ttft, .99),
            "itl_p50": pct(gaps, .5), "itl_p99": pct(gaps, .99), "e2e_p50": pct(e2e, .5),
            "out_tok_s": sum(r.out_len for r in self.done if r.finish >= (s["t0"] or w.warmup_s)) / span,
            "gen_tok_s_window": s["gen_tokens_window"] / span,
            "timeouts_per_min": s["timeouts"] / span * 60,
            "errors_per_min": (s["timeouts"] + s["rejected"] + s["too_long"]) / span * 60, "rejected_per_s": s["rejected"] / span,
            "wasted_share": s["wasted_tokens"] / max(1, s["gen_tokens_window"]),
            "gen_tok_s": s["gen_tokens"] / max(1e-9, self.now),
            "prompt_mean": sum(r.prompt_len for r in done) / max(1, len(done)),
            "prompt_p99": pct(sorted(r.prompt_len for r in done), .99),
            "output_mean": sum(r.out_len for r in done) / max(1, len(done)),
            "gpu_util": min(1.0, s["busy_s"] / span),
            "gpu_mem_used_gb": plan["budget_gb"], "gpu_mem_total_gb": self.e.gpu_memory_gb,
            "running": s["running_area"] / span, "waiting": s["waiting_area"] / span,
            "kv_usage": s["kv_area"] / span, "preemptions": s["preemptions"],
            "preempt_per_min": s["preemptions"] / span * 60,
            "prefix_hit_rate": s["hit_tokens"] / max(1, s["queried_tokens"]),
            "prompt_tok_s": s["prompt_tokens_computed"] / max(1e-9, self.now),
            "mean_batch_tokens": s["batch_tokens_area"] / max(1e-9, s["busy_s"]),
            "requests": done[:],
        }


def simulate(engine, workload, seed=1):
    return Replica(engine, workload, seed).run()


# ---------------------------------------------------------------- command line
def fmt_s(x):
    return "   n/a" if x != x else (f"{x * 1000:5.0f} ms" if x < 1 else f"{x:6.2f} s")


def dashboard(sc, seed):
    a, b = simulate(sc.before_engine, sc.before_load, seed), simulate(sc.now_engine, sc.now_load, seed)
    rows = [("Requests per second (in)", lambda r: f"{r['rate_in']:.2f}"),
            ("TTFT p50", lambda r: fmt_s(r["ttft_p50"])), ("TTFT p99", lambda r: fmt_s(r["ttft_p99"])),
            ("Inter-token latency p50", lambda r: fmt_s(r["itl_p50"])), ("Inter-token latency p99", lambda r: fmt_s(r["itl_p99"])),
            ("Output tokens per second", lambda r: f"{r['out_tok_s']:.0f}"),
            ("Unfinished at end of window", lambda r: f"{r['unfinished']}"),
            ("Errors per minute (400/429/timeout)", lambda r: f"{r['errors_per_min']:.1f}"),
            ("GPU utilisation", lambda r: f"{r['gpu_util'] * 100:.0f}%"),
            ("GPU memory used", lambda r: f"{r['gpu_mem_used_gb']:.1f} / {r['gpu_mem_total_gb']:.0f} GB ({r['gpu_mem_used_gb'] / r['gpu_mem_total_gb'] * 100:.0f}%)")]
    print(f"Dashboard: {sc.title}\n")
    print(f"{'':36s} {sc.before_label:>20s} {sc.now_label:>20s}")
    for name, f in rows:
        print(f"{name:36s} {f(a):>20s} {f(b):>20s}")


def metrics(sc, seed):
    a, b = simulate(sc.before_engine, sc.before_load, seed), simulate(sc.now_engine, sc.now_load, seed)
    rows = [("vllm:num_requests_running (mean)", lambda r: f"{r['running']:.1f}"),
            ("vllm:num_requests_waiting (mean)", lambda r: f"{r['waiting']:.1f}"),
            ("vllm:kv_cache_usage_perc (mean)", lambda r: f"{r['kv_usage'] * 100:.0f}%"),
            ("vllm:num_preemptions (per minute)", lambda r: f"{r['preempt_per_min']:.1f}"),
            ("prefix cache hit rate (hits/queries)", lambda r: f"{r['prefix_hit_rate'] * 100:.0f}%"),
            ("prompt tokens computed per second", lambda r: f"{r['prompt_tok_s']:.0f}"),
            ("generation tokens per second", lambda r: f"{r['gen_tok_s']:.0f}"),
            ("tokens per engine step (mean)", lambda r: f"{r['mean_batch_tokens']:.0f}"),
            ("mean prompt length (tokens)", lambda r: f"{r['prompt_mean']:.0f}"),
            ("p99 prompt length (tokens)", lambda r: f"{r['prompt_p99']:.0f}"),
            ("mean output length (tokens)", lambda r: f"{r['output_mean']:.0f}")]
    print(f"Engine metrics: {sc.title}\n")
    print(f"{'':38s} {sc.before_label:>20s} {sc.now_label:>20s}")
    for name, f in rows:
        print(f"{name:38s} {f(a):>20s} {f(b):>20s}")


def startup_log(e: Engine):
    p = e.memory_plan()
    print(f"INFO  Model weights took {e.weights_gb:.2f} GB")
    print(f"INFO  Memory budget: gpu_memory_utilization={e.gpu_memory_utilization:.2f} x {e.gpu_memory_gb:.0f} GB = {p['budget_gb']:.2f} GB")
    print(f"INFO  Peak activation memory (profiled) {e.activation_gb:.2f} GB, non-torch memory {e.non_torch_gb:.2f} GB, CUDA graphs {p['graphs_gb']:.2f} GB")
    print(f"INFO  Available KV cache memory: {p['kv_gb']:.2f} GB")
    print(f"INFO  KV cache dtype {e.kv_cache_dtype}: {e.kv_bytes_per_token():,} bytes per token "
          f"(2 x {e.layers} layers x {e.kv_heads} KV heads x {e.head_dim} x {1 if e.kv_cache_dtype == 'fp8' else 2} bytes)")
    print(f"INFO  GPU KV cache size: {p['kv_tokens']:,} tokens ({p['blocks']:,} blocks of {e.block_size})")
    print(f"INFO  Maximum concurrency for {e.max_model_len:,} tokens per request: {p['kv_tokens'] / e.max_model_len:.2f}x")
    print(f"INFO  CUDA graphs: {'disabled (enforce_eager=True)' if e.enforce_eager else 'captured'}")


def config(sc):
    print(f"Engine arguments ({sc.before_label} -> {sc.now_label}); changed values are marked *\n")
    for f in fields(Engine):
        x, y = getattr(sc.before_engine, f.name), getattr(sc.now_engine, f.name)
        print(f"{'*' if x != y else ' '} {f.name:24s} {str(x):>10s} {str(y):>10s}")


def requests_sample(sc, seed, n=8):
    r = simulate(sc.now_engine, sc.now_load, seed)
    print(f"Request log sample ({sc.now_label}): prompt tokens, cached tokens on admission, output tokens, TTFT, preemptions\n")
    print(f"{'id':>6s} {'prompt':>7s} {'cached':>7s} {'output':>7s} {'TTFT':>9s} {'preempted':>9s}")
    for q in r["requests"][len(r["requests"]) // 2:][:n]:
        print(f"{q.rid:6d} {q.prompt_len:7d} {q.hit_tokens:7d} {q.out_len:7d} {fmt_s(q.first_token - q.arrival):>9s} {q.preemptions:9d}")


def sweep(sc, seed, levels=(4, 8, 12, 16, 24, 32, 48, 64, 128)):
    print(f"Closed-loop concurrency sweep: {sc.title}\n")
    print("useful = output tokens per second of requests that completed; generated = all tokens the GPU produced\n")
    print(f"{'clients':>7s} {'useful tok/s':>12s} {'generated':>10s} {'TTFT p50':>10s} {'running':>8s} {'KV used':>8s} {'preempt/min':>11s} {'timeouts/min':>12s} {'429/s':>6s}")
    for c in levels:
        r = simulate(sc.now_engine, replace(sc.now_load, clients=c), seed)
        print(f"{c:7d} {r['out_tok_s']:12.0f} {r['gen_tok_s_window']:10.0f} {fmt_s(r['ttft_p50']):>10s} {r['running']:8.1f} {r['kv_usage'] * 100:7.0f}% "
              f"{r['preempt_per_min']:11.1f} {r['timeouts_per_min']:12.1f} {r['rejected_per_s']:6.1f}")


def apply_overrides(sc, pairs):
    for pair in pairs:
        key, _, val = pair.partition("=")
        for obj in ("now_engine", "now_load"):
            target = getattr(sc, obj)
            ftypes = {f.name: f.type for f in fields(target)}
            if key in ftypes:
                old = getattr(target, key)
                new = val.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(val)
                setattr(sc, obj, replace(target, **{key: new}))
                break
        else:
            raise SystemExit(f"Unknown setting {key!r}. Engine settings: {[f.name for f in fields(Engine)]}; "
                             f"workload settings: {[f.name for f in fields(Workload)]}")


def main(argv):
    import scenarios
    args, sets, seed = [], [], 1
    it = iter(argv)
    for a in it:
        if a == "--set":
            sets.append(next(it))
        elif a == "--seed":
            seed = int(next(it))
        else:
            args.append(a)
    if not args or args[0] == "list":
        print("Scenarios:"); [print(f"  {k:12s} {v(1).title}") for k, v in scenarios.SCENARIOS.items()]; return
    if args[0] not in scenarios.SCENARIOS:
        raise SystemExit(f"Unknown scenario {args[0]!r}; try: list")
    sc = scenarios.SCENARIOS[args[0]](seed)
    apply_overrides(sc, sets)
    cmd = args[1] if len(args) > 1 else "brief"
    if cmd == "brief": print(sc.brief)
    elif cmd == "dashboard": dashboard(sc, seed)
    elif cmd == "metrics": metrics(sc, seed)
    elif cmd == "changes": print("Recent changes:\n" + "\n".join(f"  - {c}" for c in sc.changes))
    elif cmd == "config": config(sc)
    elif cmd == "startup-log": startup_log(sc.now_engine)
    elif cmd == "requests": requests_sample(sc, seed)
    elif cmd == "sweep": sweep(sc, seed)
    elif cmd == "reveal": print(sc.answer)
    else: raise SystemExit(f"Unknown command {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
