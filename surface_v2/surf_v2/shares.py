"""The opponents' test reads on Alpine in shares: convlines.evaluate --net none over consecutive chunks of the test set
(the analysis joins the shares in chunk order), several processes at a time, each share its own process, log and read
files. A share whose records all exist already is skipped, so a job that ended early is simply run again.

    python -m surf_v2.shares --opponents "plain PM" "correlated PM" --shots SHOTS --save READS --logs LOGS --jobs 60
    python -m surf_v2.shares --bposd --opponents "min-LLR BP-OSD" "BP 60 BP-OSD" --shots SHOTS --save READS --logs LOGS
        --jobs 8 --gpus 0,1 --size 5        (BP-OSD on its subset; processes spread over the job's GPUs)"""
import argparse
import json
import os
import subprocess
import sys
import time

from convlines import evaluate, testset

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=cfg["test"]["set"])
    ap.add_argument("--d", default="all")
    ap.add_argument("--opponents", nargs="+", required=True)
    ap.add_argument("--bposd", action="store_true", help="the BP-OSD subset (config test.bposd_chunks)")
    ap.add_argument("--chunks", default="", help="every d: these chunks (dry runs); default the config's")
    ap.add_argument("--shots", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--size", type=int, default=10, help="chunks per share")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpus", default="", help="comma list: CUDA_VISIBLE_DEVICES of the processes, round robin")
    a = ap.parse_args()
    ds = cfg["distances"] if a.d == "all" else [int(x) for x in a.d.split(",")]
    gpus = [g for g in a.gpus.split(",") if g]
    os.makedirs(a.save, exist_ok=True)
    os.makedirs(a.logs, exist_ok=True)
    todo = []
    for d in ds:
        ch = testset.parse_range(a.chunks or (cfg["test"]["bposd_chunks"] if a.bposd else cfg["test"]["chunks"])[str(d)])
        for i in range(0, len(ch), a.size):
            part = ch[i:i + a.size]
            spec = f"{part[0]}-{part[-1]}"
            assert part == list(range(part[0], part[-1] + 1))
            done = [os.path.exists(os.path.join(a.save, f"{a.set}_d{d}_{evaluate.KEYS[o]}_{spec}.json")) for o in a.opponents]
            if all(done):
                continue
            assert not any(done), f"d{d} {spec}: some of the share's reads exist, not all: remove them and run again"
            todo.append((d, spec))
    print(f"{len(todo)} shares to read with {a.jobs} processes", flush=True)
    run, bad, k = {}, [], 0
    while todo or run:
        while todo and len(run) < a.jobs:
            d, spec = todo.pop(0)
            env = dict(os.environ, OMP_NUM_THREADS="1")
            if gpus:
                env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
            k += 1
            log = open(os.path.join(a.logs, f"{a.set}_d{d}_{'_'.join(evaluate.KEYS[o] for o in a.opponents)}_{spec}.log"), "w")
            p = subprocess.Popen([sys.executable, "-m", "convlines.evaluate", "--set", a.set, "--d", str(d), "--chunks", spec,
                                  "--shots", a.shots, "--net", "none", "--opponents", *a.opponents, "--save", a.save],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            run[p] = (d, spec, log, time.time())
        time.sleep(2)
        for p in [p for p in run if p.poll() is not None]:
            d, spec, log, t0 = run.pop(p)
            log.close()
            print(f"d{d} chunks {spec}: exit {p.returncode} in {time.time() - t0:.0f} s", flush=True)
            if p.returncode:
                bad.append((d, spec))
    print(f"failed shares: {bad}" if bad else "every share read", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
