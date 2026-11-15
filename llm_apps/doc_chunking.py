"""Document processing for RAG: fixed-size chunks versus structure-aware chunks.

Parsers (PDF to Markdown, OCR, layout models) give you text with structure: headings, tables, lists.
Fixed-size chunking ignores that and cuts tables and sections in half. Structure-aware chunking:
  - splits at headings and carries the heading path into every chunk ("Plans > Team plan"),
  - keeps a table together and also emits one self-contained sentence per row,
  - never cuts inside a list item.
"""

DOC = """# Pricing guide
Prices are in US dollars and exclude local taxes.

## Plans
### Individual plan
Full access to every course. Billed monthly or yearly.

### Team plan
For companies with five or more learners. Includes invoices and an admin view.

| Plan | Monthly | Yearly | Seats |
|---|---|---|---|
| Individual | 19 | 190 | 1 |
| Team | 15 per seat | 150 per seat | 5 minimum |

## Refunds
- Digital courses: within 7 days if under 20% of lessons were opened.
- Duplicate purchases: at any time after review.
"""


def fixed_chunks(text, size=120):
    return [text[i:i + size] for i in range(0, len(text), size)]


def structured_chunks(text):
    chunks, path, buf, table = [], [], [], []

    def flush():
        if buf:
            chunks.append(f"[{' > '.join(path)}] " + " ".join(buf))
            buf.clear()

    def flush_table():
        if not table:
            return
        header = [c.strip() for c in table[0].strip("|").split("|")]
        rows = [[c.strip() for c in r.strip("|").split("|")] for r in table[2:]]
        chunks.append(f"[{' > '.join(path)}] table: " + " / ".join(table))
        for r in rows:
            pairs = ", ".join(f"{h}: {v}" for h, v in zip(header[1:], r[1:]))
            chunks.append(f"[{' > '.join(path)}] {header[0]} {r[0]}: {pairs}.")
        table.clear()

    for line in text.splitlines():
        if line.startswith("#"):
            flush(); flush_table()
            level = len(line) - len(line.lstrip("#"))
            path[:] = path[: level - 1] + [line.lstrip("# ").strip()]
        elif line.startswith("|"):
            flush(); table.append(line)
        elif line.startswith("- "):
            flush(); flush_table()
            chunks.append(f"[{' > '.join(path)}] {line[2:]}")
        elif line.strip():
            flush_table(); buf.append(line.strip())
    flush(); flush_table()
    return chunks


def best_chunk(chunks, query):
    q = set(query.lower().replace("?", "").split())
    return max(chunks, key=lambda c: (len(q & set(c.lower().replace(",", " ").replace(".", " ").replace(":", " ").split())), -len(c)))


if __name__ == "__main__":
    fixed, smart = fixed_chunks(DOC), structured_chunks(DOC)
    print(f"fixed-size: {len(fixed)} chunks, structure-aware: {len(smart)} chunks\n")
    q = "team plan yearly price per seat"
    print("query:", q)
    print("fixed-size best   ->", repr(best_chunk(fixed, q)))
    print("structure best    ->", best_chunk(smart, q))
    print("\ntable row chunks:")
    for c in smart:
        if "Plan Team:" in c or "Plan Individual:" in c:
            print("  ", c)
