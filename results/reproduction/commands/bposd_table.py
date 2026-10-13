import json, csv, glob, os, sys
raw = sys.argv[1]; repo = sys.argv[2]
reg = {(int(r["d"]), r["vs"]): r for f in ("s2_test_eval.csv", "s3_test_eval.csv") for r in csv.DictReader(open(f"{repo}/results/{f}"))}
names = {"min-LLR BP-OSD": "bposd_minllr", "BP 60 BP-OSD": "bposd_bp60"}
print("| d | shots read here | ours | opponent | failures here | McNemar p | verdict | registered: shots, ours, opponent, p, verdict |")
print("|---|---|---|---|---|---|---|---|")
for d in (7, 9, 11, 13, 15):
    for fn in (f"eval_bposd_d{d}.json", f"eval_bposd_sub_d{d}.json"):
        f = f"{raw}/out/{fn}"
        if not os.path.exists(f): continue
        r = json.load(open(f)); o = r["ours"]
        for n, c in r["cells"].items():
            k = reg[(d, names[n])]
            print(f"| {d} | {r['shots']:,} ({r['chunks']}) | {o['failures']:,} | {n} | {c['fail_b']:,} | {c['mcnemar_p']:.3g} | {c['verdict']} | "
                  f"{int(k['shots']):,}, {int(k['fail_a']):,}, {int(k['fail_b']):,}, {float(k['mcnemar_p']):.3g}, {k['verdict']} |")
        break
