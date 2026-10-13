"""A1's ramp reruns (revision/REVISION_PLAN.md 2.1, "A head that does not train"): from the main runs' not-trained
checks (at 2e7 samples), which ramp runs to start. Per code and seed: every arm whose check says not trained gets one
ramp run, and the seed's deployed arm gets one (once, however many of the seed's arms fail); nothing for a seed whose
arms all trained; never a second ramp run (a ramp run that stays at its constant guess is reported, not rerun).
Prints the sbatch commands of the ramp runs not yet started (their run directory absent).
    python -m a1.ramp --code surface [--seeds 1 2]"""
import argparse
import json
import os

from a1 import seeds as a1seeds

# Each seed's GPU model and time limit, as in a1/jobs/submit_main.sh (revision/A1_READY.md section 5).
PART = {("surface", 1): "-p ah200 --gres=gpu:h200:1 -c 16 --mem=64G -t 05:00:00",
        ("surface", 2): "-p aa100 --gres=gpu:a100_80gb:1 -c 12 --mem=64G -t 06:00:00",
        ("bb72", 1): "-p ah200 --gres=gpu:h200_2g.35gb:1 -c 8 --mem=48G -t 04:25:00",
        ("bb72", 2): "-p artxpro6000 --gres=gpu:rtx_pro_6000_1g.24gb:1 -c 8 --mem=48G -t 06:05:00"}


def check_of(code, run):
    if code == "surface":
        p = os.path.join(os.environ["ICMP_CKPT"], run, "train_check.json")
        return json.load(open(p)) if os.path.exists(p) else None
    p = os.path.join(os.environ["BB_CKPT"], run, "log.json")
    if os.path.exists(p):
        return json.load(open(p)).get("train_check")
    p = os.path.join(os.environ["BB_CKPT"], run, "last.pt")
    if os.path.exists(p):
        import torch
        return torch.load(p, map_location="cpu", weights_only=False)["run"].get("train_check")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", choices=["surface", "bb72"], required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(a1seeds.SEEDS))
    a = ap.parse_args()
    arms = a1seeds.SURFACE_ARMS if a.code == "surface" else a1seeds.BB_ARMS
    dep = a1seeds.DEPLOYED[a.code]
    ck_dir = os.environ["ICMP_CKPT" if a.code == "surface" else "BB_CKPT"]
    cmd = "train_surface.txt" if a.code == "surface" else "train_bb72.txt"
    out = {"code": a.code, "seeds": {}}
    for s in a.seeds:
        checks = {arm: check_of(a.code, f"a1_{arm}_s{s}") for arm in arms}
        failed = [arm for arm, c in checks.items() if c is not None and not c["trained"]]
        pending = [arm for arm, c in checks.items() if c is None]
        ramp = ([dep] if failed else []) + [arm for arm in failed if arm != dep]
        todo = [arm for arm in ramp if not os.path.exists(os.path.join(ck_dir, f"a1ramp_{arm}_s{s}"))]
        out["seeds"][s] = {"failed": failed, "checks_pending": pending, "ramp": ramp, "to_start": todo}
        for arm in todo:
            print(f"sbatch {PART[(a.code, s)]} -J a1ramp_{arm}_s{s} --export=ALL,CMD={cmd},ARM={arm},S={s},RAMP=1 "
                  f"code/ising_compare/a1/jobs/job.sbatch")
    print("RAMP", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
