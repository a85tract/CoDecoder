"""G4 (D7; row E2): the Triton engine at d = 7 with the deployed d = 7 network, timed as the GPU term at d = 9-17 is.
The studies deployed d = 7 as a resident CUDA kernel (timing/d7); the paper's 17.2 us for the Triton engine at d = 7 was
a development measurement. Here the engine of surface_v2/timing/deployed.py (FROZEN, built from the released weights'
deploy tables in fp32 on the CPU) runs surface_v2/timing/gpu_term.py's production loop `frozen()`, unchanged, on the
200,000 shots of the timing set at d = 7 (VIEW), in fresh processes; the statistic is the GPU term's, the slowest of
five accepted processes.

Acceptance, as gpu_term's for d = 9-17: the outputs (the first 2,000 shots) equal, bit for bit, this engine's own c
computed once beforehand one shot at a time (`ref`, timing.deployed.Logits' loop for the Triton engine); no other
compute process on the GPU before or after; every SM-clock sample of the timed part at least 97 % of the maximum,
throttle reasons none or idle. The resident kernel's c is not the reference: another kernel rounds differently.
    python -m speed_rt.triton7 ref --view VIEW --out MASKS                 -> MASKS/triton7_d7.npz
    python -m speed_rt.triton7 time --view VIEW --ref MASKS/triton7_d7.npz --proc K --core C --out SESSION
-> SESSION/triton7_d7_p<K>.json (written once)."""
import argparse
import json
import os
import time

import numpy as np
import torch

NREF = 2000


def shots(view):
    from convlines import testset
    sh = [testset.load(os.path.join(view, testset.name("val", 7, j)), 7) for j in range(20)]
    return np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("ref", "time"))
    ap.add_argument("--view", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--proc", type=int, default=0)
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="dry runs only: the first N shots")
    a = ap.parse_args()
    from convlines import decoder, provenance
    from timing import deployed, gpu_term
    apps0 = gpu_term.smi_apps()                 # before this process has a CUDA context: gpu_term.own_pids finds the line that appears
    name = decoder.NETWORKS[7]
    det, obs = shots(a.view)
    t, cfg = deployed.tables(7)
    eng = deployed.engine(deployed.to_cuda(t), cfg)
    os.makedirs(a.out, exist_ok=True)
    if a.what == "ref":
        x = torch.from_numpy(np.ascontiguousarray(det[:NREF])).cuda()
        c = torch.empty(NREF, dtype=torch.float32, device="cuda")
        with torch.inference_mode():
            for i in range(NREF):
                c[i:i + 1].copy_(eng(x[i:i + 1])[0])
        out = os.path.join(a.out, "triton7_d7.npz")
        np.savez(out, c=c.cpu().numpy(), d=7, network=name, logits="triton7",
                 shots_sha256=deployed.shots_sha256(det, obs))
        print(f"G4 ref: c of the first {NREF} shots by the Triton engine at d = 7 -> {out}", flush=True)
        return
    core = a.core if a.core >= 0 else max(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {core})
    torch.set_num_threads(1)
    z = np.load(a.ref)
    assert str(z["shots_sha256"]) == deployed.shots_sha256(det, obs), f"{a.ref} belongs to other shots"
    res = dict(item="G4", network=name, network_sha256=decoder.weights_sha256(name), engine="Triton FROZEN",
               code_digest=provenance.code_digest(), d=7, proc=a.proc, core=core, pid=os.getpid(),
               gpu=torch.cuda.get_device_name(), torch=torch.__version__, limit=a.limit or None,
               load_start=open("/proc/loadavg").read().split()[:3], apps_before=apps0,
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    res["own_pids"] = gpu_term.own_pids(apps0)
    d = np.ascontiguousarray(det[:a.limit] if a.limit else det)
    r, got = gpu_term.frozen(eng, torch.from_numpy(d).cuda(), core, "off", False)
    res.update(r, shots_set="the timing set at d = 7, chunks 0-19 in order" + (f" (first {a.limit})" if a.limit else ""),
               outputs_checked=len(got), outputs_equal_ref=bool(np.array_equal(got.view(np.uint32),
                                                                               z["c"][:len(got)].view(np.uint32))))
    res.update(load_end=open("/proc/loadavg").read().split()[:3], end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    torch.cuda.synchronize()
    res["apps_after"] = gpu_term.smi_apps()
    c = res["clocks"]
    chk = dict(clocks=c.get("sm_min_frac") is not None and c["sm_min_frac"] >= 0.97 and
               set(c["throttle_seen"]) <= {"0x0000000000000000", "0x0000000000000001"},
               gpu_alone=all(gpu_term.pid_of(x) in res["own_pids"] for x in apps0 + res["apps_after"]),
               outputs_equal_ref=res["outputs_equal_ref"])
    res.update(acceptance=chk, accepted=all(chk.values()))
    with open(os.path.join(a.out, f"triton7_d7_p{a.proc}.json"), "x") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"G4 triton7 p{a.proc}: per-shot mean {res['per_shot_mean']:.3f} us over {res['shots']} shots; checks {chk}; "
          f"accepted {res['accepted']}", flush=True)


if __name__ == "__main__":
    main()
