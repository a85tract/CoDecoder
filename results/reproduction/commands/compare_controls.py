import json, csv, os, sys
raw = sys.argv[1]; repo = sys.argv[2]
reg = {(int(r["d"]), r["opponent"]): r for r in csv.DictReader(open(f"{repo}/results/controls_test_eval.csv"))}
sha = {(int(r["d"]), r["model"]): r for r in csv.DictReader(open(f"{repo}/results/controls_decisions.csv"))}
print("| d | ours | control | failures: this run / registered | only ours / only control: this run (registered) | p | verdict | decisions sha256 equal | observables sha256 equal |")
print("|---|---|---|---|---|---|---|---|---|")
for d in (7, 9, 11, 13, 15):
    f = f"{raw}/ctrl_d{d}.json"
    if not os.path.exists(f): continue
    r = json.load(open(f))
    for n, m in (("Ising M1 + correlated PM", "M1"), ("Ising M4 + correlated PM", "M4")):
        c = r["cells"][n]; k = reg[(d, f"{m}_corr")]; h = sha[(d, m)]
        print(f"| {d} | {r['ours']['failures']:,} | {n} | {c['fail_b']:,} / {int(k['fail_b']):,} | {c['only_a']:,} / {c['only_b']:,} ({int(k['only_a']):,} / {int(k['only_b']):,}) | "
              f"{c['mcnemar_p']:.3g} | {c['verdict']} / {k['verdict']} | {c['decisions_sha256'] == h['decisions_sha256']} | {r['obs_sha256'] == h['obs_sha256']} |")
