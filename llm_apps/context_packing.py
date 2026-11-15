"""Context engineering: fit the most useful material into a fixed token budget.

Each candidate item has a priority class and a relevance score. Pinned items (system rules, the user's
question) always go in; the rest are packed greedily by value per token, older chat turns are summarised
when they do not fit, and the final order puts the most important material at the start and end of the
prompt, where models attend best. Token counts use a rough 4-characters-per-token rule.
"""

def tokens(text):
    return max(1, (len(text) + 3) // 4)


ITEMS = [
    {"id": "system", "kind": "pinned", "text": "You are a support assistant. Answer only from the provided notes and cite them as [n]." * 2},
    {"id": "question", "kind": "pinned", "text": "Can I get a refund on a digital course I bought 10 days ago?"},
    {"id": "note-refunds", "kind": "doc", "score": 0.91, "text": "Refund policy: digital courses can be refunded within 7 days of purchase if under 20% of lessons were opened. " * 3},
    {"id": "note-exceptions", "kind": "doc", "score": 0.84, "text": "Exceptions: duplicate purchases and billing errors are refunded at any time after review. " * 3},
    {"id": "note-shipping", "kind": "doc", "score": 0.22, "text": "Shipping: physical orders ship within two working days to most countries. " * 4},
    {"id": "note-pricing", "kind": "doc", "score": 0.35, "text": "Pricing: plans are billed monthly or yearly; yearly saves about two months. " * 4},
    {"id": "turn-1", "kind": "history", "score": 0.40, "text": "User: hi, I bought the inference course last week. Assistant: Thanks! How can I help with it today? " * 3},
    {"id": "turn-2", "kind": "history", "score": 0.55, "text": "User: the videos are fine but it is not what I expected. Assistant: I am sorry to hear that. " * 3},
    {"id": "tool-schema", "kind": "tool", "score": 0.60, "text": '{"name":"create_refund_ticket","parameters":{"order_id":"string","reason":"string"}}'},
]


def summarise(item):
    """Stand-in for a model-written summary: keep the first sentence."""
    first = item["text"].split(". ")[0].rstrip(".") + "."
    return {**item, "id": item["id"] + "(summary)", "text": first}


def pack(items, budget):
    chosen = [i for i in items if i["kind"] == "pinned"]
    used = sum(tokens(i["text"]) for i in chosen)
    rest = sorted((i for i in items if i["kind"] != "pinned"), key=lambda i: (-i["score"] / tokens(i["text"]), i["id"]))
    dropped = []
    for i in rest:
        t = tokens(i["text"])
        if used + t <= budget:
            chosen.append(i); used += t
        elif i["kind"] == "history" and used + tokens(summarise(i)["text"]) <= budget:
            s = summarise(i); chosen.append(s); used += tokens(s["text"])
        else:
            dropped.append(i["id"])
    # Order: system first, question last, the highest-scoring evidence right before the question.
    body = sorted((i for i in chosen if i["kind"] != "pinned"), key=lambda i: (i["kind"] != "history", i.get("score", 0)))
    ordered = [i for i in chosen if i["id"] == "system"] + body + [i for i in chosen if i["id"] == "question"]
    return ordered, used, dropped


if __name__ == "__main__":
    total = sum(tokens(i["text"]) for i in ITEMS)
    print(f"all candidates: {total} tokens")
    for budget in (400, 250):
        ordered, used, dropped = pack(ITEMS, budget)
        print(f"\nbudget {budget}: used {used}")
        print("  order  :", " > ".join(i["id"] for i in ordered))
        print("  dropped:", ", ".join(dropped) or "none")
