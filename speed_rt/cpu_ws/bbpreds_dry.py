"""DRY-RUN-ONLY fabricator of the GPU-pass prediction files that the BB CPU terms read (bb.stage3 cputerm, bb.stage4_gross cputerm).
The CPU host has no GPU, so for a dry run the routing read (r12 on BB72, g2 on the gross code) is replaced by files in the format
bb.route.infer writes: $BB_PREDS/<code>_p<p>_<split><j:04d>_net-<tag>.npz with pred, margin, fail, sig, meta. The margin routes exactly
the registered fraction of the dry shots, the ones of the LARGEST syndrome weight (a proxy for hard shots), at every frozen tau of the
cells that read the file: BB72-3S/3F (p = 0.3 %, 1.98 %), BB72-2S/2F (p = 0.2 %, 0.24 %); the gross file serves G-4R (0.96 %) and
G-4F (1.99 %) at their two nested taus. pred is zero, fail = (pred != obs).any(1); sig is the registered engine record (BB72: the
cell policy's engine with the analysed GPU, as retest/replay/bb_replay.py unpack does; gross: cell.json "engine"), so the cells'
signature checks pass; meta says "dry synthetic". These are NOT network reads: never use them for a registered number.
    python bbpreds_dry.py --tree CHECKOUT --draw DRAWDIR [--role bb72|gross|all] [--preds DIR]
CHECKOUT: the public tree (bb/bb72/ising_compare, bb/gross_stage4/ising_compare). DRAWDIR: a dry draw (speed_rt/bb/draw.sh BEACON
DRAWDIR dry) with shots/ and a seedcheck.json that says dry and ok. Files go to DRAWDIR/preds (= BB_PREDS) unless --preds; an
existing file is never replaced (nothing is written when any target exists). Standard library and numpy only."""
import argparse
import json
import os
import sys

import numpy as np

# registered routed shots / test shots (bb72: data/bb/stage3/analysis*.log; gross: data/bb/stage4_gross/results.json)
REG72 = {0.003: (3960, 200_000), 0.002: (721, 300_000)}
REGG = {"G-4R": (1447, 150_000), "G-4F": (2983, 150_000)}


def rng(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def sets_bb72(tree):
    """One set per (p, tag): the cells BB72-3S/3F (p 0.003) and 2S/2F (p 0.002) read r12 at one tau per p."""
    d = os.path.join(tree, "bb", "bb72", "ising_compare", "data", "bb")
    cells = json.load(open(os.path.join(d, "stage3", "cells.json")))
    allc = {**cells["cells"], **json.load(open(os.path.join(d, "stage3", "amendment_S.json")))["cells"]}
    out = {}
    for name, c in sorted(allc.items()):
        if c["tau"] is None:                                   # BB72-3 and BB72-2 route nothing
            continue
        sp = cells["splits"][c["split"]]
        sig = dict(json.load(open(os.path.join(d, "policies", c["policy"])))["engine"], gpu=cells["gpu"])
        if sig["net_sha256"] != c["net_sha256"]:
            raise SystemExit(f"{name}: the policy's engine is not the cell's network")
        s = out.setdefault((sp["p"], c["tag"]), dict(code=sp["code"], p=sp["p"], split="test", chunks=rng(sp["chunks"]),
                                                    tag=c["tag"], sig=sig, cuts={}))
        if s["sig"] != sig:
            raise SystemExit(f"{name}: two engine records for one read")
        s["cuts"].setdefault(c["tau"], dict(names=[], reg=REG72[sp["p"]]))["names"].append(name)
    return list(out.values())


def sets_gross(tree):
    """The gross read serves G-4R and G-4F, each at its own tau (G-4R's routed shots are a subset of G-4F's)."""
    c = json.load(open(os.path.join(tree, "bb", "gross_stage4", "ising_compare", "data", "bb", "stage4_gross", "cell.json")))
    s = dict(code=c["code"], p=c["p"], split=c["split"], chunks=rng(c["chunks"]), tag=c["tag"], sig=c["engine"], cuts={})
    for name, reg in REGG.items():
        s["cuts"].setdefault(c["cells"][name]["tau"], dict(names=[], reg=reg))["names"].append(name)
    return [s]


def margins(w, cuts):
    """w: syndrome weights of the whole set; cuts: [(tau, k)] by rising tau. Rank 0 = largest weight (ties: the earlier shot). The k
    hardest shots get margin < tau; the shots between two cuts lie inside [tau_i-1, tau_i); the rest is above the last tau.
    Returns (float32 margin, shot indices in rank order)."""
    n = len(w)
    order = np.argsort(-w, kind="stable")
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n)
    taus, ks = [t for t, _ in cuts], [k for _, k in cuts]
    if any(b <= a for a, b in zip(taus, taus[1:])) or any(b < a for a, b in zip(ks, ks[1:])):
        raise SystemExit(f"taus {taus} and routed counts {ks} are not nested")
    m = np.empty(n)
    start = 0
    for lo, hi, k in zip([taus[0] - 2.0] + taus[:-1], taus, ks):
        sel = (rank >= start) & (rank < k)
        m[sel] = lo + (hi - lo) * (0.2 + 0.6 * (rank[sel] - start + 0.5) / max(k - start, 1))
        start = k
    sel = rank >= start
    m[sel] = taus[-1] + 1.0 + 4.0 * (rank[sel] - start + 0.5) / max(n - start, 1)
    return m.astype(np.float32), order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--draw", required=True)
    ap.add_argument("--role", choices=["bb72", "gross", "all"], default="all")
    ap.add_argument("--preds", default=None)
    a = ap.parse_args()
    sc = json.load(open(os.path.join(a.draw, "seedcheck.json")))
    if sc.get("dry") is not True or not sc.get("ok"):
        raise SystemExit(f"{a.draw}: not a dry draw that passed its seed check: refusing to fabricate predictions")
    preds = a.preds or os.path.join(a.draw, "preds")
    sets = (sets_bb72(a.tree) if a.role in ("bb72", "all") else []) + (sets_gross(a.tree) if a.role in ("gross", "all") else [])
    shot = lambda s, j: os.path.join(a.draw, "shots", f"{s['code']}_p{s['p']}_{s['split']}{j:04d}.npz")
    out = lambda s, j: os.path.join(preds, f"{s['code']}_p{s['p']}_{s['split']}{j:04d}_net-{s['tag']}.npz")
    missing = [shot(s, j) for s in sets for j in s["chunks"] if not os.path.exists(shot(s, j))]
    exists = [out(s, j) for s in sets for j in s["chunks"] if os.path.exists(out(s, j))]
    if missing or exists:
        raise SystemExit(f"{len(missing)} shot files missing (first {missing[:1]}); {len(exists)} prediction files exist "
                         f"(first {exists[:1]}): nothing written")
    os.makedirs(preds, exist_ok=True)
    for s in sets:
        ws, fails, K = [], [], 0
        for j in s["chunks"]:
            with np.load(shot(s, j)) as z:
                n, D, K = int(z["n"]), int(z["D"]), int(z["k"])
                ws.append(np.unpackbits(z["det"], axis=1, count=D)[:n].sum(1, dtype=np.int64))
                fails.append(np.unpackbits(z["obs"], axis=1, count=K)[:n].any(1))      # pred is zero, so fail = obs.any
        w = np.concatenate(ws)
        cuts = sorted(s["cuts"].items())
        ks = [int(round(len(w) * c["reg"][0] / c["reg"][1])) for _, c in cuts]
        m, order = margins(w, [(t, k) for (t, _), k in zip(cuts, ks)])
        for (t, _), k in zip(cuts, ks):                                      # the consumer's own test, as it writes it
            if not np.array_equal(np.flatnonzero(m < t), np.sort(order[:k])):
                raise SystemExit(f"internal check failed at tau {t}")
        meta = json.dumps({"dry": "dry synthetic", "what": "bbpreds_dry.py: margin routes the largest-syndrome-weight shots, pred zero; "
                           "not a network read", "cells": sorted(x for _, c in cuts for x in c["names"]),
                           "routed": {str(t): k for (t, _), k in zip(cuts, ks)}})
        sig = json.dumps(s["sig"], sort_keys=True)
        off = 0
        for j, fail in zip(s["chunks"], fails):
            n = len(fail)
            mj = m[off:off + n]
            off += n
            with open(out(s, j), "xb") as fh:                                # x: never replaces a file
                np.savez(fh, pred=np.zeros((n, K), np.uint8), margin=mj, fail=fail, sig=sig, meta=meta)
            print(f"{os.path.basename(out(s, j))} shots {n} routed " + " ".join(f"{'/'.join(c['names'])}: {int((mj < t).sum())}"
                                                                                 for t, c in cuts), flush=True)
        print(f"# {s['code']} p {s['p']} {s['tag']}: {len(w)} shots, " + ", ".join(
            f"{'/'.join(c['names'])} routed {k} ({100 * k / len(w):.2f} %, registered {100 * c['reg'][0] / c['reg'][1]:.2f} %)"
            for (t, c), k in zip(cuts, ks)), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
