import csv, re, sys, glob
repo = sys.argv[1]; raw = sys.argv[2]
reg = {(int(r["d"]), r["vs"]): int(r["fail_b"]) for f in ("s2_test_eval.csv", "s3_test_eval.csv") for r in csv.DictReader(open(f"{repo}/results/{f}"))}
print("| d | part | failures: this run / registered | difference |")
print("|---|---|---|---|")
for f in sorted(glob.glob(f"{raw}/logs/trt_eval_d*.log"), key=lambda x: int(re.search(r"d(\d+)", x).group(1))):
    for line in open(f):
        m = re.match(r"^(M[14]_\S+) d(\d+): (\d+) failures", line)
        if m:
            part, d, n = m.group(1), int(m.group(2)), int(m.group(3)); r = reg.get((d, part))
            print(f"| {d} | {part} | {n:,} / {r:,} | {n - r:+d} |" if r is not None else f"| {d} | {part} | {n:,} / (no registered row) | |")
