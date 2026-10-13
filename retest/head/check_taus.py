"""The thresholds' check of the output-head re-test (RETEST.md section head). A1 froze the rule, not the numbers: tau_q is
the numpy quantile of each network's confidence on 200,000 calibration shots, computed inside a1.analyze. The re-test
feeds the analysis A1's own calibration reads, so pair_curves.csv must carry A1's taus, digit for digit.
    python check_taus.py CODE PAIR_CURVES.csv TAUS.csv [OUT.json]
TAUS.csv (code, arm, seed, q, tau) holds the original pair_curves.csv's rows (126 rows, 18 arm/seed networks x 7 q; q = 0
has tau = -inf). Compares the tau strings of the (arm, seed, q) rows; exit 1 on a missing, extra or different one.
Standard library only."""
import csv
import json
import sys


def main():
    code, new_path, ref_path = sys.argv[1:4]
    ref = {(r["arm"], r["seed"], r["q"]): r["tau"] for r in csv.DictReader(open(ref_path)) if r["code"] == code}
    new = {(r["arm"], r["seed"], r["q"]): r["tau"] for r in csv.DictReader(open(new_path))}
    missing = sorted(set(ref) - set(new))
    extra = sorted(set(new) - set(ref))
    differ = sorted((k, ref[k], new[k]) for k in set(ref) & set(new) if ref[k] != new[k])
    nets = {(a, s) for a, s, q in ref}
    res = {"code": code, "networks": len(nets), "rows_expected": len(ref), "rows_equal": len(ref) - len(missing) - len(differ),
           "missing": [list(k) for k in missing], "extra": [list(k) for k in extra],
           "differ": [[list(k), a, b] for k, a, b in differ], "ok": not (missing or extra or differ)}
    if len(sys.argv) > 4:
        json.dump(res, open(sys.argv[4], "w"), indent=1)
    print("TAU CHECK", json.dumps({k: v for k, v in res.items() if k not in ("missing", "extra", "differ")}),
          "missing", len(missing), "extra", len(extra), "differ", len(differ), flush=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
