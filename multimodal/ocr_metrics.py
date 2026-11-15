"""Measuring document AI: character and word error rate, field accuracy, and table cell accuracy.

OCR and document parsers are evaluated against a hand-checked reference. Three levels:
  - CER / WER: edit distance at character or word level, divided by reference length
  - field accuracy: did we extract the fields the application needs (invoice number, total, date)?
  - table cells: fraction of reference cells reproduced exactly, in the right row and column
A low CER can still hide a wrong total, which is why field-level metrics matter more for applications.
"""


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return edit_distance(ref, hyp) / len(ref)


def wer(ref, hyp):
    return edit_distance(ref.split(), hyp.split()) / len(ref.split())


REF = "Invoice INV-20931 dated 2026-10-04. Total due: 1,284.50 EUR. Payment within 30 days."
HYPS = {
    "parser A": "Invoice INV-20931 dated 2026-10-04. Total due: 1,284.50 EUR. Payment within 3O days.",
    "parser B": "Invoice INV-20931 dated 2026-10-04. Total due: 1,234.50 EUR. Payment within 30 days.",
    "parser C": "Invoice INV 20931 dated 2026 10 04 Total due 1,284.50 EUR Payment within 30 days",
}
FIELDS = {"invoice": r"INV-\d+", "date": r"\d{4}-\d{2}-\d{2}", "total": r"\d[\d,]*\.\d{2}"}

REF_TABLE = [["Item", "Qty", "Price"], ["GPU hours", "120", "960.00"], ["Storage", "1", "24.50"], ["Support", "1", "300.00"]]
HYP_TABLE = [["Item", "Qty", "Price"], ["GPU hours", "120", "960.00"], ["Storage", "1", "300.00"], ["Support", "1", "24.50"]]


if __name__ == "__main__":
    import re
    ref_fields = {k: re.search(p, REF).group(0) for k, p in FIELDS.items()}
    print(f"{'':<9} {'CER':>6} {'WER':>6}  fields correct")
    for name, hyp in HYPS.items():
        got = {k: (m.group(0) if (m := re.search(p, hyp)) else None) for k, p in FIELDS.items()}
        ok = [k for k in FIELDS if got[k] == ref_fields[k]]
        print(f"{name:<9} {cer(REF, hyp):>6.3f} {wer(REF, hyp):>6.3f}  {len(ok)}/3 {sorted(set(FIELDS) - set(ok)) or ''}")

    cells = [(r, c) for r in range(1, len(REF_TABLE)) for c in range(3)]
    right = sum(REF_TABLE[r][c] == HYP_TABLE[r][c] for r, c in cells)
    text_ref = sorted(x for row in REF_TABLE for x in row)
    text_hyp = sorted(x for row in HYP_TABLE for x in row)
    print(f"\ntable: same set of cell strings: {text_ref == text_hyp};  cells in the right place: {right}/{len(cells)}")
