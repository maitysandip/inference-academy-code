"""Build a small synthetic ticket-classification dataset. No downloads, fully deterministic."""
import json, random

random.seed(7)
CATEGORIES = {
    "billing":  ["I was charged twice for my subscription", "my invoice shows the wrong amount", "I need a refund for last month", "the discount code did not apply at checkout", "my card was declined but money left my account", "I was billed after cancelling my plan", "the tax on my invoice looks incorrect"],
    "login":    ["I cannot sign in to my account", "the password reset email never arrives", "two-factor codes are not working", "my account is locked after too many attempts", "I forgot which email I registered with", "the login page keeps redirecting me", "I never received the verification link"],
    "shipping": ["my order has not arrived yet", "the tracking page has not updated in days", "the package arrived damaged", "I received the wrong item", "the delivery address needs to be changed", "the courier marked my parcel as delivered but it is missing", "part of my order is missing from the box"],
    "bug":      ["the app crashes when I open settings", "the export button does nothing", "pages load very slowly today", "the search returns no results for anything", "the dashboard shows an error message", "uploads fail halfway through", "the mobile app freezes on the home screen"],
    "feature":  ["it would be great to have dark mode", "please add support for CSV import", "can you add a weekly summary email", "I would like to share reports with my team", "an option to change the language would help", "an API for exporting data would be useful", "a keyboard shortcut for search would save time"],
}
URGENT = ["This is urgent, ", "Right now ", "This has blocked our whole team: ", "Our customers are affected: "]
CALM = ["When you have time, ", "No rush, but ", "Just a note: ", "A small thing: "]
NORMAL = ["", "Hello, ", "Hi team, "]
EXTRA = ["", "", " (order #{n})", " (ticket ref {n})", " on my {dev}", " since {day}"]
DEV = ["phone", "laptop", "tablet", "work computer"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "yesterday"]

def make(n):
    rows = []
    while len(rows) < n:
        cat = random.choice(list(CATEGORIES))
        base = random.choice(CATEGORIES[cat])
        pri = random.choices(["high", "medium", "low"], weights=[3, 4, 3])[0]
        if cat == "feature" and pri == "high": pri = "medium"          # feature requests are never high priority
        prefix = random.choice(URGENT if pri == "high" else CALM if pri == "low" else NORMAL)
        extra = random.choice(EXTRA).format(n=random.randint(1000, 9999), dev=random.choice(DEV), day=random.choice(DAY))
        text = prefix + base + extra + random.choice([".", ".", " please.", " thanks."])
        text = text[0].upper() + text[1:]
        rows.append({"ticket": text, "label": {"category": cat, "priority": pri}})
    return rows

pool = make(700)
seen, uniq = set(), []
for r in pool:                       # drop exact duplicates before splitting
    if r["ticket"] not in seen: seen.add(r["ticket"]); uniq.append(r)
train, test = uniq[:480], uniq[480:580]
assert not ({r["ticket"] for r in train} & {r["ticket"] for r in test})   # no train/test leakage
json.dump({"train": train, "test": test}, open("tickets.json", "w"))
print("train", len(train), "test", len(test))
print(train[0]); print(test[0])
