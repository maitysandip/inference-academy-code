"""Incident scenarios for incident_sim.py.

SPOILERS: each scenario below states its cause. Work the incident from the commands first
(brief, dashboard, changes, config, startup-log, metrics, requests, sweep), write down your
diagnosis, and only then read this file or run the `reveal` command.
"""
import random
from dataclasses import dataclass, field, replace
from incident_sim import Engine, Workload


@dataclass
class Scenario:
    title: str
    brief: str
    before_label: str
    now_label: str
    before_engine: Engine
    before_load: Workload
    now_engine: Engine
    now_load: Workload
    changes: list = field(default_factory=list)
    answer: str = ""


CHAT = Workload(rate=0.33, duration_s=900, prompt_mean=850, output_mean=220, system_prompts=3, system_prompt_len=1536)


def healthy(seed=1):
    return Scenario(
        title="Warm-up: a healthy replica",
        brief=("WARM-UP. Nothing is wrong. Before you run anything, predict the decode step time, the size of the\n"
               "KV cache in tokens and the median TTFT from the engine settings (config). Then check your numbers\n"
               "with dashboard, startup-log, metrics and sweep."),
        before_label="yesterday 14:00", now_label="today 14:00",
        before_engine=Engine(), before_load=CHAT, now_engine=Engine(), now_load=CHAT,
        changes=["no changes in the last 7 days"],
        answer="Nothing to find: this replica is healthy. Compare your predictions with the lesson's worked numbers.")


def ttft(seed=1):
    return Scenario(
        title="TTFT went from about 400 ms to about 4 s",
        brief=("PAGE 14:20. Chat API, one model replica per GPU. Median time to first token was about 400 ms\n"
               "yesterday and is about 4 s now. No errors. Traffic looks normal. Find the cause, prove it, fix it.\n"
               "Start with:  dashboard, then changes, then whatever the evidence points to."),
        before_label="yesterday 14:00", now_label="today 14:00",
        before_engine=Engine(), before_load=CHAT,
        now_engine=Engine(), now_load=replace(CHAT, header_position="start"),
        changes=["11:05 prompt-service v2.14: chat template now includes request metadata (time, request id) for audit logging",
                 "10:40 node image patched (kernel 6.8.0-49 -> 6.8.0-51); GPU driver unchanged",
                 "09:15 client SDK 3.2.0 rolled out to the web app (retry back-off tuned)",
                 "yesterday 17:30 Grafana dashboards moved to a new folder"],
        answer=("Cause: prompt-service v2.14 puts the per-request metadata at the START of the prompt. Every prompt now\n"
                "differs from its first token, so prefix caching finds nothing: the hit rate falls from about 63% to about 1%.\n"
                "Prompt length barely changed (the same 24 tokens moved), but each request now prefills about 2,400 tokens\n"
                "instead of about 900. Prefill work rises about 2.7x, the GPU runs close to saturation, requests queue,\n"
                "and queueing turns a 2.7x prefill cost into an 8x median TTFT and a 20x p99.\n"
                "Proof: metrics shows the hit rate collapse and prompt tokens computed per second up about 2.7x while\n"
                "requests per second and prompt length are flat. The requests sample shows 0 cached tokens.\n"
                "Fix: move dynamic content to the END of the prompt (--set header_position=end), or roll back v2.14.\n"
                "Prevent: alert on the prefix cache hit rate; review prompt-template changes for cache impact."))


RAG = Workload(rate=0.26, duration_s=1200, warmup_s=180, prompt_mean=1800, prompt_sigma=0.5, output_mean=400,
               output_sigma=0.5, system_prompts=0)


def memory(seed=1):
    return Scenario(
        title="GPU memory at 90% while the weights are 35 GB of 48 GB",
        brief=("TICKET from the platform team: 'The RAG model's GPU sits at 90% memory all day, but the weights are\n"
               "only 35 GB of 48 GB. We think vLLM leaks memory. At the afternoon peak, p99 TTFT goes past 20 s.\n"
               "Should we restart it nightly or buy bigger GPUs?' Decide what is really happening and what to change.\n"
               "Start with:  dashboard, then startup-log."),
        before_label="02:00 (quiet)", now_label="14:00 (peak)",
        before_engine=Engine(), before_load=replace(RAG, rate=0.08),
        now_engine=Engine(), now_load=RAG,
        changes=["no deploys in the last 7 days",
                 "RAG retrieval raised from 6 to 8 chunks last month (prompts about 1,800 tokens on average)",
                 "nightly restart cron proposed in ticket OPS-2211 (not applied)"],
        answer=("Cause: nothing leaks. vLLM reserves gpu_memory_utilization x total memory (0.90 x 48 = 43.2 GB) at\n"
                "startup, so 90% is the design, at 2 a.m. and at peak alike. The real problem is what is left for the\n"
                "KV cache: 43.2 - 35 weights - 1.6 activations - 0.5 non-torch - 0.6 CUDA graphs = 5.5 GB, or about\n"
                "21,000 tokens. With RAG prompts near 2,200 tokens including output, only about 9 requests fit at\n"
                "once. At peak more arrive than that pool can serve: they wait, and some running ones are preempted.\n"
                "Fixes, cheapest first: kv_cache_dtype=fp8 (doubles the pool to about 42,000 tokens);\n"
                "gpu_memory_utilization=0.95 if nothing else shares the GPU (+2.4 GB, about +9,000 tokens);\n"
                "fewer retrieved chunks; then a second replica or two GPUs with tensor parallelism. A nightly restart\n"
                "changes nothing; lowering max_model_len does not enlarge the pool either.\n"
                "Check: --set kv_cache_dtype=fp8 (and validate quality), then --set gpu_memory_utilization=0.95."))


LOADTEST = Workload(clients=16, duration_s=600, warmup_s=120, prompt_mean=1200, prompt_sigma=0.5, output_mean=400,
                    output_sigma=0.4, max_tokens=1024, system_prompts=0, timeout_s=90)


def concurrency(seed=1):
    return Scenario(
        title="Throughput falls about 40% as concurrency rises",
        brief=("LOAD TEST before launch. Same replica, closed-loop clients with no think time, the production SDK's\n"
               "settings. Successful output tokens per second climb to a peak and then FALL as you add clients: about\n"
               "40% below the peak by 40 clients, worse beyond. Explain the curve, find what is responsible, and say\n"
               "what you would change before launch.\n"
               "Start with:  sweep, then dashboard and metrics, then changes and startup-log."),
        before_label="16 clients", now_label="40 clients",
        before_engine=Engine(), before_load=LOADTEST,
        now_engine=Engine(), now_load=replace(LOADTEST, clients=40),
        changes=["engine arguments are the defaults: max_num_seqs=256, max_num_batched_tokens=8192",
                 "load generator: closed loop, prompts about 1,200 tokens, outputs about 400, no shared prefix",
                 "client SDK settings copied from production: 90 s request timeout, retry at once on timeout",
                 "no gateway or rate limiter in front of the replica in the test environment"],
        answer=("Cause: congestion collapse. The KV cache holds about 21,000 tokens, so only 12 to 15 requests of this\n"
                "size can run at once (startup-log). Past that, extra clients just queue inside the engine. When the\n"
                "queue wait plus generation time passes the SDK's 90 s timeout, the client disconnects, the engine\n"
                "aborts the request, and the client sends it again. The GPU stays busy (generated tokens per second\n"
                "hardly moves) but a growing share of its work belongs to requests nobody will receive, so useful\n"
                "throughput falls: about 31% down at 32 clients, about 41% at 40, 62% at 48.\n"
                "Proof: sweep shows 'generated' flat while 'useful' falls, timeouts per minute rising from 0, KV usage\n"
                "pinned in the 90s and preemptions appearing once the pool is full.\n"
                "Fix: admission control in front of the engine, sized to what it can really run (--set gateway_limit=12):\n"
                "excess requests get a fast 429 and back off, so useful throughput stays at the peak. Then raise the\n"
                "peak itself: kv_cache_dtype=fp8 doubles the pool (with gateway_limit=24 it holds about 255 tok/s).\n"
                "Lowering max_num_seqs alone does not help: the queue just moves and still times out."))


# ---- blind capstone: a random fault from a pool ----
BLIND_FAULTS = [
    ("prefix", "a prompt-template change put per-request metadata at the start of the prompt",
     lambda e, w: (e, replace(w, header_position="start")),
     "the prefix cache hit rate falls to about 0%, prompt tokens computed per second jump and TTFT rises, "
     "while requests per second and prompt length stay flat",
     "--set header_position=end (or roll back the template)"),
    ("memutil", "gpu_memory_utilization was lowered from 0.90 to 0.82 to make room for a sidecar on the same GPU",
     lambda e, w: (replace(e, gpu_memory_utilization=0.82), w),
     "GPU memory used drops to 82% (a clue hiding in plain sight), startup-log shows a much smaller KV cache, "
     "KV usage rises, preemptions appear and TTFT p99 jumps",
     "--set gpu_memory_utilization=0.90 and move the sidecar to another GPU"),
    ("eager", "enforce_eager=True was left on after a debugging session, disabling CUDA graphs",
     lambda e, w: (replace(e, enforce_eager=True), w),
     "inter-token latency rises by a few milliseconds per token at every load while TTFT barely moves; "
     "config shows enforce_eager changed and startup-log says CUDA graphs are disabled",
     "--set enforce_eager=false"),
    ("seqs", "the 'tidy flags' PR changed max_num_seqs from 256 to 8",
     lambda e, w: (replace(e, max_num_seqs=8), w),
     "requests running is pinned at 8 while KV usage stays low and requests wait; config shows max_num_seqs changed",
     "--set max_num_seqs=256"),
    ("traffic", "a new enterprise tenant sends prompts about twice as long as before",
     lambda e, w: (e, replace(w, prompt_mean=w.prompt_mean * 2)),
     "mean and p99 prompt length jump and prompt tokens per second rise; config shows nothing relevant",
     "no engine flag fixes extra demand: rate-limit or route the tenant, or add capacity"),
]


def blind(seed=1):
    # Every run of five consecutive seeds (5-9, 10-14, ...) covers all five faults, in a shuffled order.
    order = random.Random(seed // len(BLIND_FAULTS)).sample(range(len(BLIND_FAULTS)), len(BLIND_FAULTS))
    name, cause, inject, evidence, fix = BLIND_FAULTS[order[seed % len(BLIND_FAULTS)]]
    base_e = Engine(kv_cache_dtype="fp8", max_num_batched_tokens=2048)
    base_w = replace(CHAT, rate=0.40)
    now_e, now_w = inject(base_e, base_w)
    changes = ["09:00 routine: on-call handover",
               "10:30 engine image rebuilt from the same tag (base image security patches)",
               "11:45 config repo: engine arguments reformatted and merged (PR 'tidy flags')",
               "13:10 sales: onboarding call with a new enterprise tenant"]
    return Scenario(
        title=f"Blind incident (seed {seed})",
        brief=("PAGE. Users report the assistant 'feels slower' since late morning. You do not know what changed.\n"
               "Use every command, write a timeline and a diagnosis with evidence, apply a fix with --set, show the\n"
               "dashboard recovering, then write the postmortem. Run `reveal` only after you have written it."),
        before_label="yesterday", now_label="today",
        before_engine=base_e, before_load=base_w, now_engine=now_e, now_load=now_w,
        changes=changes,
        answer=f"Fault [{name}]: {cause}.\nEvidence you should have found: {evidence}.\nFix: {fix}")


SCENARIOS = {"healthy": healthy, "ttft": ttft, "memory": memory, "concurrency": concurrency, "blind": blind}
