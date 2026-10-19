"""SPEED-RT stage 3 (STAGE3_batched.md section 3): the CPU session's windows, in blocks, from the decision passes' routed
counts and stage 2's raw per-shot times (the expected lengths decide only the packing and the budget check).

    python -m speed_rt.s3.plan3 --decide DIR --out PLAN.json        DIR: the decision passes' decide_<point>.json"""
import argparse
import json
import os

from speed_rt.s3.cpu3 import FIRST, ITEMS

# stage 2's raw per-row times in us (speed_rt/s2/surface/results, speed_rt/s2/bb/results): every shot, or a routed shot
US_ALL = {("surf-d7", "pm"): 7.876, ("surf-d9", "pm"): 17.418, ("surf-d11", "pm"): 34.597, ("surf-d13", "pm"): 60.741,
          ("surf-d15", "pm"): 106.037, ("surf-d17", "pm"): 160.640,
          ("surf-d7", "cpm"): 18.280, ("surf-d9", "cpm"): 42.335, ("surf-d11", "cpm"): 91.331,
          ("surf-d13", "cpm"): 167.623, ("surf-d15", "cpm"): 372.678, ("surf-d17", "cpm"): 560.359,
          ("bb72-p2", "sector"): 1320, ("bb72-p3", "sector"): 2190, ("gross-p4", "sector"): 84500,
          ("bb72-p2", "full"): 35000, ("bb72-p3", "full"): 56600, ("gross-p4", "full"): 2525000}
US_ROUTED = {("surf-d7", "cpm"): 33.5, ("surf-d9", "cpm"): 63.4, ("surf-d11", "cpm"): 100.0, ("surf-d13", "cpm"): 200.0,
             ("surf-d15", "cpm"): 488.3, ("surf-d17", "cpm"): 778.9,
             ("bb72-p2", "sector"): 7350, ("bb72-p3", "sector"): 11050, ("gross-p4", "sector"): 1379000,
             ("bb72-p2", "full"): 89200, ("bb72-p3", "full"): 162500, ("gross-p4", "full"): 14142000}
START_S = 4.0        # an item's start-up in a process (shots, decoder, warm rows) before its barrier
WINDOW_S = 6.0       # a window's reference segments, clock probes and launch
REPS = (1, 2, 3)
BLOCKS = [("a", "surface", lambda it, n: it.startswith("surf")),
          ("b", "BB72 compositions and sector Relay-BP", lambda it, n: it.startswith("bb72") and not it.endswith(".full")),
          ("c", "BB72 full-DEM Relay-BP alone, N = 8", lambda it, n: it.startswith("bb72") and it.endswith(".full") and n == 8),
          ("d", "BB72 full-DEM Relay-BP alone, N = 1", lambda it, n: it.startswith("bb72") and it.endswith(".full") and n == 1),
          ("e", "gross compositions, and its baselines at N = 8",
           lambda it, n: it.startswith("gross") and (it.endswith((".bal", ".acc")) or n == 8)),
          ("f", "gross baselines at N = 1", lambda it, n: it.startswith("gross") and not it.endswith((".bal", ".acc")) and n == 1)]


def rows(item, dec):
    """(rows the item decodes, expected seconds of one process's whole share at N = 1)."""
    pt, kind, sel, first = ITEMS[item]
    d = dec[pt]
    n_all = d["shots"] if first is None else min(first, d["shots"])
    if sel == "all":
        r = n_all
        us = US_ALL[(pt, kind)]
    else:
        r = d["taus"][sel]["routed"] if first is None else None
        us = US_ROUTED[(pt, kind)]
    return r, r * us / 1e6


def job(items, k, n, rep, wid):
    return dict(id=f"{wid}.k{k}" if n > 1 else f"{items[0]}.N1.r{rep}", base=f"{wid}.k{k}" if n > 1 else f"{items[0]}.N1.r{rep}",
                items=items, N=n, k=k, rep=rep, part=",".join(items), side="group" if n > 1 else
                ("ours" if items[0].endswith((".ours", ".bal", ".acc")) else "baseline"),
                argv=["{py}", "-m", "speed_rt.s3.cpu3", "--items", ",".join(items), "--k", str(k), "--n", str(n),
                      "--rep", str(rep), "--group", "{out}/.." if n > 1 else "{out}", "--out", "{out}"],
                cwd="{code}", env={"PYTHONPATH": "{code}:{code}/surface_v2", "PYTHONDONTWRITEBYTECODE": "1",
                                   "S3_DATA": "{data}", "QND_RELAY": "{relay}"})


def plan(dec):
    runs = []                                         # (block, item, N, rep, expected seconds of the run)
    for item in sorted(ITEMS):
        pt = ITEMS[item][0]
        if pt not in dec:
            continue
        r, s1 = rows(item, dec)
        if r == 0:
            continue                                  # nothing routed: C(N) = 0, no process
        for n in (1, 8):
            for rep in REPS:
                blk = next(b for b, _, f in BLOCKS if f(item, n))
                runs.append((blk, item, n, rep, START_S + s1 / n))
    blocks = []
    for b, name, _ in BLOCKS:
        eights = [[x for x in runs if x[0] == b and x[2] == 8 and x[3] == rep] for rep in REPS]
        ones = sorted([x for x in runs if x[0] == b and x[2] == 1], key=lambda x: (-x[4], x[3], x[1]))
        n1 = [ones[i:i + 8] for i in range(0, len(ones), 8)]
        wins = []
        for i in range(max(len(REPS), len(n1))):
            if i < len(REPS) and eights[i]:
                g = sorted(eights[i], key=lambda x: (x[4], x[1]))      # shortest first: barriers wait behind short items
                wid = f"{b}{REPS[i]}_N8"
                wins.append(dict(id=wid, rep=REPS[i], expected_s=round(sum(x[4] for x in g) + WINDOW_S, 1),
                                 jobs=[job([x[1] for x in g], k, 8, REPS[i], wid) for k in range(8)]))
            if i < len(n1):
                grp = n1[i] if i % 2 == 0 else n1[i][::-1]            # sides alternate across slots window by window
                wins.append(dict(id=f"{b}_N1_{i}", rep=None, expected_s=round(max(x[4] for x in grp) + WINDOW_S, 1),
                                 jobs=[job([x[1]], 0, 1, x[3], None) for x in grp]))
        blocks.append(dict(block=b, name=name, windows=wins, expected_s=round(sum(w["expected_s"] for w in wins), 1)))
    return dict(what="SPEED-RT stage 3 CPU plan (speed_rt/s3/plan3.py)", first=FIRST, blocks=blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dec = {}
    for f in sorted(os.listdir(a.decide)):
        if f.startswith("decide_") and f.endswith(".json"):
            d = json.load(open(os.path.join(a.decide, f)))
            dec[d["point"]] = d
    p = plan(dec)
    with open(a.out, "x") as fh:
        json.dump(p, fh, indent=1)
    for b in p["blocks"]:
        print(f"block {b['block']} ({b['name']}): {len(b['windows'])} windows, expected {b['expected_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
