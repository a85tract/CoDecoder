"""The composite of this reproduction from its own timing records (raw/timing_out, raw/cpu_out), with the registered
rule (timing/composite.py): (a) the slowest of five GPU-term processes, (b) b_node x ratio from cpu_term convert,
(c) at d = 7 the largest handoff of five processes, at d = 9-15 max(1.1 us, the router's estimate); the opponents'
registered Grace-basis times from results/. It ran on the pod from the release's root; its raw/timing_out records were
not archived, so it cannot be rerun from this tree (REPRODUCTION.md section 4 gives its output)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from timing import composite as C  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")


def main():
    reg = {r[0]: r for r in C.rows()}
    print("d   (a) GPU [reg]        (b) node  ratio   (b) Grace [reg]     (c) [reg]          composite [reg]      fastest opponent      margin  cells")
    allhold = True
    for d, name in C.KEY.items() | {15: "d15_b5_r782"}.items():
        a = max(json.load(open(f))["per_shot_mean"] for f in glob.glob(f"{RAW}/timing_out/{name}_p[0-4].json"))
        h = [json.load(open(f)) for f in sorted(glob.glob(f"{RAW}/timing_out/{name}_handoff_p*.json"))]
        c = max(x["handoff_us"] for x in h) if d == 7 else max(1.1, h[0]["handoff_estimate_us"])
        b = json.load(open(f"{RAW}/cpu_out/b_d{d}.json"))
        comp = a + b["b_grace_us"] + c
        _, ra, rb_node, rratio, rc, rcomp, opp, _, _, _, _ = reg[d]
        cells = sum(comp < us for us in opp.values())
        allhold &= cells == 6
        fast = min(opp, key=opp.get)
        print(f"{d:<3d} {a:7.3f} [{ra:7.3f}]  {b['b_node_us']:8.3f}  {b['ratio']:.4f}  {b['b_grace_us']:7.3f} [{rb_node * rratio:7.3f}]  "
              f"{c:5.3f} [{rc:5.3f}]  {comp:8.3f} [{rcomp:8.3f}]  {fast:20s} {opp[fast] - comp:7.3f}  {cells}/6 hold")
    print("all 30 speed cells hold with the reproduced inputs:", allhold)


if __name__ == "__main__":
    main()
