import json, csv, sys
root = sys.argv[1]; repo = sys.argv[2]
reg = {}
for f in ("s2_test_eval.csv", "s3_test_eval.csv"):
    for r in csv.DictReader(open(f"{repo}/results/{f}")):
        reg[(int(r["d"]), r["vs"])] = r
names = {"plain PM": "pm", "correlated PM": "pm_corr", "Ising M1 + PM": "M1_torch", "Ising M4 + PM": "M4_torch",
         "min-LLR BP-OSD": "bposd_minllr", "BP 60 BP-OSD": "bposd_bp60"}
print("| d | shots | ours (repro / registered) | routed (repro) | opponent | failures (repro / registered) | McNemar p (repro / registered) | verdict |")
print("|---|---|---|---|---|---|---|---|")
for d in (7, 9, 11, 13, 15):
    for fn in (f"eval_d{d}.json", f"eval_bposd_d{d}.json"):
        try: r = json.load(open(f"{root}/{fn}"))
        except FileNotFoundError: continue
        o = r["ours"]
        for n, c in r["cells"].items():
            k = reg[(d, names[n])]
            print(f"| {d} | {r['shots']:,} | {o['failures']:,} / {int(k['fail_a']):,} | {o['routed']:,} ({100*o['routed_fraction']:.2f} %) | {n} | "
                  f"{c['fail_b']:,} / {int(k['fail_b']):,} | {c['mcnemar_p']:.3g} / {float(k['mcnemar_p']):.3g} | {c['verdict']} / {k['verdict']} |")
