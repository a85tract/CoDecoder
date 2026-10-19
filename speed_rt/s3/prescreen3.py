"""Stage 3's pod pre-screen (STAGE3_batched.md section 2): speed_rt.gpu_prescreen unchanged (stage 1's rule: five GPU
checks, CPU steal at most 1 % over 5 s, the container's CPU quota), except that on a pod whose container quota is
below 20 CPUs the quota limit is 12 CPUs (the orchestrator's decision for this stage, recorded beside the record).

    python -m speed_rt.s3.prescreen3 --out DIR      -> DIR/prescreen.json and DIR/quota_rule.json; exit 4 when it fails"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REL)
sys.path.insert(0, os.path.join(REL, "surface_v2"))

from speed_rt import gpu_prescreen as G  # noqa: E402


def quota():
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(per)
    except OSError:
        return None


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    q = quota()
    q = float(len(os.sched_getaffinity(0))) if q is None else q
    rule = dict(quota_cpus=q, limit=20.0 if q >= 20 else 12.0,
                why="stage 3: a pod with fewer than 20 CPUs is screened at 12 (the orchestrator's decision D1); otherwise stage 1's 20")
    G.LIMITS["quota_cpus"] = rule["limit"]
    try:
        G.main()
    finally:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "quota_rule.json"), "w") as fh:
            json.dump(rule, fh, indent=1)
