"""INT8 per-row and grouped 4-bit weight quantisation, and quantised matvec."""
def quantize_rows(W):
    Q, S = [], []
    for row in W:
        s = max(abs(v) for v in row) / 127 or 1.0
        Q.append([round(v / s) for v in row]); S.append(s)
    return Q, S

def matvec_fp(W, x): return [sum(a * b for a, b in zip(r, x)) for r in W]
def matvec_q8(Q, S, x): return [s * sum(a * b for a, b in zip(r, x)) for r, s in zip(Q, S)]

def q4_group(W, g=32):
    out = []
    for row in W:
        r = []
        for i in range(0, len(row), g):
            grp = row[i:i + g]; s = max(abs(v) for v in grp) / 7 or 1.0
            r += [round(v / s) * s for v in grp]
        out.append(r)
    return out
