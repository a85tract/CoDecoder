"""ARBCP, d = 13: the routed shots of the re-test's C5 ("the pair": d13_b5_r775 + CS13) on Cascade's two compiled paths,
from the re-test's confirmation read itself (light numpy, no GPU). That read saved Cascade's float32 logit on both
paths (B1 = compiled@1 on every shot; BQ5 = compiled@16 over the queue of the shots SB routes under C5's tau1, in shot
order, the last batch zero-padded; Bb = compiled@16 over every shot, Cascade alone in batched mode), so nothing is run
again; the decisions are the read's own (dec), checked against z > 0.

  python cp_d13x.py --read RETEST/out/d13/conf --lock cascade/d13/out/locked_d13.json --out DIR

Per p (0.6 %, 0.35 %), as sv_common.assemble builds C5 and C2: single = S1 (routing by |c| of S1) with B1 on the
routed shots, Cascade alone B1; batched = SB (routing by |c| of SB) with BQ5 on the routed shots, Cascade alone Bb.
Writes DIR/d13_<p>.npz (per shot routed in either mode: index, observable, S1 / SB logit and decision, B1 / BQ5 / Bb
logit and decision) and DIR/d13_<p>.json (checks, kept-shot counts per mode)."""
import argparse
import json
import os

import numpy as np

from surface import sv_common as sc

TAU1 = {0.006: 5.242445659637451, 0.0035: 8.111059150695802}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read", required=True)
    ap.add_argument("--lock", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = json.load(open(a.lock))
    for p in (0.006, 0.0035):
        tau1 = float(L["p"][str(p)]["chosen"]["C5"]["tau1"])
        if tau1 != TAU1[p]:
            raise SystemExit(f"C5's tau1 at {p}: {tau1}, not {TAU1[p]}")
        o = sc.load_read(a.read, p, ["S1", "SB", "B1", "Bb", "BQ5"])
        obs = o["obs"].astype(np.uint8).ravel()
        n = len(obs)
        S1, SB, B1, Bb, BQ = o["S1"], o["SB"], o["B1"], o["Bb"], o["BQ5"]
        r1, rB = sc.routed(S1["c"], tau1), sc.routed(SB["c"], tau1)
        rec = {"p": p, "tau1": tau1, "shots": int(n), "chunks": list(o["chunks"]), "routed_single": int(r1.sum()),
               "routed_batched": int(rB.sum())}
        u32 = lambda x: np.ascontiguousarray(x, np.float32).view(np.uint32)   # noqa: E731
        rec["checks"] = {
            "S1_equals_SB_bitwise": bool(np.array_equal(u32(S1["c"]), u32(SB["c"]))
                                         and np.array_equal(S1["dec"], SB["dec"])),
            "S_dec_is_c_gt_0": bool(np.array_equal((S1["c"] > 0).astype(np.uint8), S1["dec"])
                                    and np.array_equal((SB["c"] > 0).astype(np.uint8), SB["dec"])),
            "B1_dec_is_z_gt_0": bool(np.array_equal((B1["z"] > 0).astype(np.uint8), B1["dec"])),
            "Bb_dec_is_z_gt_0": bool(np.array_equal((Bb["z"] > 0).astype(np.uint8), Bb["dec"])),
            "BQ5_exactly_on_routed": bool(np.array_equal(BQ["dec"] != 255, rB)),
            "BQ5_dec_is_z_gt_0": bool(np.array_equal((BQ["z"][rB] > 0).astype(np.uint8), BQ["dec"][rB])),
            "BQ5_vs_Bb_on_routed": {"decisions_differ": int((BQ["dec"][rB] != Bb["dec"][rB]).sum()),
                                    "logit_bits_differ": int((u32(BQ["z"][rB]) != u32(Bb["z"][rB])).sum())},
            "B1_vs_BQ5_on_routed": {"decisions_differ": int((B1["dec"][rB] != BQ["dec"][rB]).sum()),
                                    "logit_bits_differ": int((u32(B1["z"][rB]) != u32(BQ["z"][rB])).sum())}}
        fS1, fSB = S1["dec"] != obs, SB["dec"] != obs
        fB1, fBb = B1["dec"] != obs, Bb["dec"] != obs
        rec["modes"] = {}
        for mode, r, fS, fA in (("single", r1, fS1, fB1), ("batched", rB, fSB, fBb)):
            k = ~r
            rec["modes"][mode] = {"S_fail_kept": int((fS & k).sum()), "alone_fail": int(fA.sum()),
                                  "kept_S_only": int((fS & ~fA & k).sum()), "kept_C_only": int((~fS & fA & k).sum()),
                                  "kept_both": int((fS & fA & k).sum())}
        idx = np.nonzero(r1 | rB)[0]
        np.savez_compressed(os.path.join(a.out, f"d13_{p}.npz"), idx=idx, obs=obs[idx], r1=r1[idx], rB=rB[idx],
                            c1=S1["c"][idx], d1=S1["dec"][idx], cB=SB["c"][idx], dB=SB["dec"][idx],
                            zB1=B1["z"][idx], dB1=B1["dec"][idx], zQ=BQ["z"][idx], dQ=BQ["dec"][idx],
                            zBb=Bb["z"][idx], dBb=Bb["dec"][idx])
        json.dump(rec, open(os.path.join(a.out, f"d13_{p}.json"), "w"), indent=1)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
