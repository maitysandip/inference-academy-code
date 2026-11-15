"""Evaluating agent trajectories, not just final answers.

A trajectory is the list of steps an agent took: tool calls with arguments, results, and the final answer.
Two runs can give the same final answer while one of them took a forbidden action, called the wrong tool,
or looped. We score each trajectory on:
  - outcome:     is the final answer right?
  - required:    were the required tool calls made (with the right key arguments)?
  - forbidden:   was any forbidden action taken? (a hard fail regardless of outcome)
  - order:       were order constraints respected (e.g. look up before refund)?
  - efficiency:  steps used versus a reference step count
"""

TASK = {
    "question": "Refund order 1182 if it is eligible.",
    "answer_contains": "refunded",
    "required": [("get_order", {"order_id": 1182}), ("issue_refund", {"order_id": 1182})],
    "forbidden": [("issue_refund", {"order_id": 1183}), ("delete_order", None)],
    "order": [("get_order", "issue_refund")],
    "reference_steps": 2,
}

RUNS = {
    "A clean": [("get_order", {"order_id": 1182}), ("issue_refund", {"order_id": 1182}), ("final", "Order 1182 was refunded.")],
    "B no lookup": [("issue_refund", {"order_id": 1182}), ("final", "Order 1182 was refunded.")],
    "C wrong order": [("get_order", {"order_id": 1182}), ("issue_refund", {"order_id": 1183}), ("issue_refund", {"order_id": 1182}), ("final", "Order 1182 was refunded.")],
    "D looping": [("get_order", {"order_id": 1182})] * 4 + [("issue_refund", {"order_id": 1182}), ("final", "Order 1182 was refunded.")],
    "E gave up": [("get_order", {"order_id": 1182}), ("final", "I could not process the refund.")],
}


def matches(step, name, args):
    if step[0] != name:
        return False
    return args is None or all(step[1].get(k) == v for k, v in args.items())


def score(traj, task):
    calls = [s for s in traj if s[0] != "final"]
    final = next((s[1] for s in traj if s[0] == "final"), "")
    outcome = task["answer_contains"] in final.lower()
    required = sum(any(matches(s, n, a) for s in calls) for n, a in task["required"]) / len(task["required"])
    forbidden = any(matches(s, n, a) for s in calls for n, a in task["forbidden"])

    def first(name):
        return next((i for i, s in enumerate(calls) if s[0] == name), None)

    order_ok = all(first(a) is not None and first(b) is not None and first(a) < first(b) for a, b in task["order"])
    efficiency = min(1.0, task["reference_steps"] / max(1, len(calls)))
    passed = outcome and required == 1.0 and not forbidden and order_ok
    return {"outcome": outcome, "required": required, "forbidden": forbidden, "order": order_ok, "efficiency": efficiency, "pass": passed}


if __name__ == "__main__":
    print(f"{'run':<14}{'outcome':>8}{'required':>9}{'forbidden':>10}{'order':>7}{'effic.':>7}{'PASS':>6}")
    for name, traj in RUNS.items():
        r = score(traj, TASK)
        print(f"{name:<14}{str(r['outcome']):>8}{r['required']:>9.2f}{str(r['forbidden']):>10}{str(r['order']):>7}{r['efficiency']:>7.2f}{str(r['pass']):>6}")
    outcome_only = sum(score(t, TASK)["outcome"] for t in RUNS.values())
    full = sum(score(t, TASK)["pass"] for t in RUNS.values())
    print(f"\nfinal-answer-only grading passes {outcome_only} of {len(RUNS)} runs; trajectory grading passes {full}.")
