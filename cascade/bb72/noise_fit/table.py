"""Aggregate cal.py's JSONL rows into the fit's table: per (point, decoder) the shots, failures, P_block with its 95 %
Wilson interval, P_L by Cascade's Eq. (1) (k = 12, R = 6) with the interval converted, the paper's digitised value
and the ratio ours / paper.

  python table.py runs/*.jsonl --json counts.json > table.md"""
import argparse
import json
import math
from collections import defaultdict

K, R = 12, 6
# Cascade Fig. 2(a), digitised from the paper's SVG (per logical qubit per round), keyed by p
PAPER = {
    "tesseract": {0.002: 8.13e-6, 0.003: 4.73e-5},
    "bposd": {0.002: 3.14e-4, 0.003: 1.30e-3},
    "relay": {0.002: 2.00e-5, 0.003: 9.50e-5},
    "cascade": {0.002: 1.04e-5, 0.003: 6.58e-5},
}
PAPER_KEY = {"tesseract": "tesseract", "bposd": "bposd", "relay_full": "relay", "relay_sector": "relay",
             "tesseract_bfs": "tesseract", "tesseract_sector": "tesseract", "bposd_sector": "bposd"}


def p_l(pb):
    """Eq. (1): P_L = (1 - (2 (1 - P_block)^(1/k) - 1)^(1/R)) / 2."""
    return (1 - (2 * (1 - pb) ** (1 / K) - 1) ** (1 / R)) / 2


def p_block(pl):
    """Inverse of Eq. (1)."""
    return 1 - ((1 + (1 - 2 * pl) ** R) / 2) ** K


def wilson(f, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    ph = f / n
    den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def load(paths):
    rows, heads = defaultdict(list), {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            key = (r["point"], r["dec"])
            if r.get("header"):
                heads.setdefault(key, []).append(r)
            else:
                rows[key].append(r)
    return rows, heads


def summarize(rows, heads):
    out = []
    for (pt, dec), rs in sorted(rows.items()):
        chunks = sorted(r["chunk"] for r in rs)
        if len(set(chunks)) != len(chunks):
            raise ValueError(f"point {pt} {dec}: a chunk was decoded twice")
        n = sum(r["shots"] for r in rs)
        f = sum(r["failures"] for r in rs)
        lo, hi = wilson(f, n)
        model, p = rs[0]["model"], rs[0]["p"]
        paper = PAPER.get(PAPER_KEY.get(dec, ""), {}).get(p)     # every noise model is compared at the same p
        pl = p_l(f / n)
        out.append(dict(point=pt, model=model, p=p, dec=dec, shots=n, failures=f, p_block=f / n, p_block_ci=[lo, hi],
                        p_l=pl, p_l_ci=[p_l(lo), p_l(hi)], paper_p_l=paper,
                        paper_p_block=p_block(paper) if paper else None, ratio=pl / paper if paper else None,
                        flagged=sum(r["flagged"] for r in rs), flagged_failures=sum(r["flagged_failures"] for r in rs),
                        decode_core_s=sum(r["secs"] for r in rs), mean_s_per_shot=sum(r["secs"] for r in rs) / n,
                        max_s=max(r["max_secs"] for r in rs), chunks=[min(chunks), max(chunks), len(chunks)],
                        seeds=[min(r["seed"] for r in rs), max(r["seed"] for r in rs)],
                        hosts=sorted({h["versions"]["host"] for h in heads.get((pt, dec), [])}),
                        fail_idx={r["chunk"]: r["fail_idx"] for r in rs}))
    return out


def paired(rows, point, dec_a, dec_b):
    """Discordance of two decoders on the chunks both read with identical shots (equal obs_digest, i.e. same seed and
    host architecture): (chunks, shots, both fail, only a fails, only b fails)."""
    a = {r["chunk"]: r for r in rows.get((point, dec_a), [])}
    b = {r["chunk"]: r for r in rows.get((point, dec_b), [])}
    common = sorted(j for j in a if j in b and a[j]["obs_digest"] == b[j]["obs_digest"])
    both = only_a = only_b = 0
    for j in common:
        fa, fb = set(a[j]["fail_idx"]), set(b[j]["fail_idx"])
        both += len(fa & fb)
        only_a += len(fa - fb)
        only_b += len(fb - fa)
    return dict(point=point, a=dec_a, b=dec_b, chunks=len(common), shots=sum(a[j]["shots"] for j in common),
                both=both, only_a=only_a, only_b=only_b)


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}e}"


def markdown(summary):
    lines = ["| point | model | p (%) | decoder | shots | failures | P_block | P_L (Eq. 1) | P_L 95 % interval | paper P_L "
             "| ours / paper | mean s/shot |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        ratio = "-" if s["ratio"] is None else f"{s['ratio']:.2f}"
        lines.append(f"| {s['point']} | {s['model']} | {100 * s['p']:.1f} | {s['dec']} | {s['shots']:,} | {s['failures']:,} "
                     f"| {fmt(s['p_block'])} | {fmt(s['p_l'])} | {fmt(s['p_l_ci'][0])} - {fmt(s['p_l_ci'][1])} "
                     f"| {fmt(s['paper_p_l'])} | {ratio} | {s['mean_s_per_shot']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, heads = load(a.paths)
    summary = summarize(rows, heads)
    print(markdown(summary))
    pairs = []
    for pt in sorted({k[0] for k in rows}):
        decs = sorted(d for (q, d) in rows if q == pt)
        for i, da in enumerate(decs):
            for db in decs[i + 1:]:
                pr = paired(rows, pt, da, db)
                if pr["chunks"]:
                    pairs.append(pr)
                    print(f"paired point {pt}: {da} vs {db}: {pr['chunks']} chunks, {pr['shots']:,} shots, both fail "
                          f"{pr['both']}, only {da} {pr['only_a']}, only {db} {pr['only_b']}")
    if a.json:
        json.dump(dict(summary=summary, paired=pairs, headers={f"{k[0]}:{k[1]}": v for k, v in heads.items()}),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
