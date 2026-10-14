"""G6: CUDA-Q QEC's BP-OSD at batch 1 on the surface code, as studies 2-4 timed it (the research hub's
ising_compare/speed/minllr_timing.py `b1`, minllr_timing3.py and minllr_timing4.py): one fresh process per (d, arm,
repetition); the release's decoder (surface_v2/convlines/opponents.py: nv-qldpc-decoder of cudaq-qec 0.8.0 on the joint
DEM of the verdict circuit, identical columns merged, OSD-CS of order 10; min-LLR with 10 iterations, or BP 60 with
final-LLR OSD initialisation); 10 untimed calls on the last 10 shots of chunk 0, then --shots host-to-host calls, call
k decoding shot k of chunk 0, perf_counter around each decode_batch of one shot. The record keeps every call's time,
mean, p50, p99 and max in us, the process CPU time over wall time (cores kept busy) and the cores visible. The
studies' statistic: the fastest of three repetitions, not converted.

Differences from the studies: the shots are the timing set's chunk 0 (VIEW: speed_rt/draw.py view), d = 7-17 in one
script, the process is pinned to the session's timed core (studies 2-4 measured 0.98-1.00 cores busy per call), and
the timed calls run inside a speed_rt.watch window, rule "throttle" (the GPU runs only the BP iterations; OSD runs on
the host), whose host and clock verdict the record carries.
    python -m speed_rt.bposd_time --d 13 --arm minllr --rep 1 --shots 200 --view VIEW --core C --out DIR
-> DIR/bposd_d13_minllr_rep1.json (written once)."""
import argparse
import json
import os
import platform
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, required=True, choices=(7, 9, 11, 13, 15, 17))
    ap.add_argument("--arm", required=True, choices=("minllr", "bp60"))
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--view", required=True, help="the timing set's view (validation-chunk file names)")
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.join(a.out, f"bposd_d{a.d}_{a.arm}_rep{a.rep}.json")
    assert not os.path.exists(out), f"{out} exists"
    os.sched_setaffinity(0, {a.core})
    import warnings
    warnings.filterwarnings("ignore")
    from speed_rt import watch
    apps0 = watch.smi_apps()           # the GPU's compute processes before this process has any CUDA context (cudaq_qec's import may create it)
    import cudaq_qec as qec
    from convlines import circuits, opponents, provenance, testset
    from timing import deployed
    det, obs = testset.load(os.path.join(a.view, testset.name("val", a.d, 0)), a.d)
    assert a.shots + 10 <= len(det), "not enough shots"
    H, O, p = opponents.check_matrices(circuits.dem(circuits.circuit(a.d), decompose=False))
    kw = {**opponents.BPOSD, **opponents.BPOSD_ARMS[a.arm]}
    dec = qec.get_decoder("nv-qldpc-decoder", H, error_rate_vec=p.tolist(), O=O, **kw)
    for i in range(10):                                          # untimed calls on shots that are not timed
        dec.decode_batch(det[len(det) - 1 - i:len(det) - i].astype(np.float64))
    new = [x for x in watch.smi_apps() if x not in apps0]
    own = {str(os.getpid())} | ({new[0].split(",")[0].strip()} if len(new) == 1 else set())
    xs = [det[k:k + 1].astype(np.float64) for k in range(a.shots)]
    ts, cs = np.empty(a.shots), np.empty(a.shots)
    w = watch.Window(a.core, "throttle")
    for k, x in enumerate(xs):
        c0, t0 = time.process_time(), time.perf_counter()
        dec.decode_batch(x)
        ts[k], cs[k] = time.perf_counter() - t0, time.process_time() - c0
    win = w.stop(own_pids=own)
    us = ts * 1e6
    gpu = __import__("subprocess").run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True).stdout.strip().splitlines()[0]
    r = dict(item="G6", d=a.d, arm=a.arm, rep=a.rep, shots=a.shots, mean_us=float(us.mean()),
             p50_us=float(np.percentile(us, 50)), p99_us=float(np.percentile(us, 99)), max_us=float(us.max()),
             calls_us=[round(float(v), 3) for v in us], cores_busy=float(cs.sum() / ts.sum()),
             cores_visible=len(os.sched_getaffinity(0)), core=a.core, pid=os.getpid(), own_pids=sorted(own),
             omp_num_threads=os.environ.get("OMP_NUM_THREADS", "unset"), settings=kw, gpu=gpu,
             versions=dict(cudaq_qec=getattr(qec, "__version__", None), python=platform.python_version()),
             shots_sha256=deployed.shots_sha256(det[:a.shots], obs[:a.shots]),
             shots_set=f"the timing set's chunk 0 of d = {a.d}, shots 0-{a.shots - 1}", code_digest=provenance.code_digest(),
             window=win, accepted=bool(win["clocks"]["ok"] and win["gpu_alone"]), end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(out, "x") as fh:
        json.dump(r, fh, indent=1)
    print(f"G6 d{a.d} {a.arm} rep {a.rep}: mean {r['mean_us']:.1f} us over {a.shots} calls, cores busy "
          f"{r['cores_busy']:.2f}; window host {win['host']['ok']} clocks {win['clocks']['ok']} alone {win['gpu_alone']}: "
          f"accepted {r['accepted']}", flush=True)


if __name__ == "__main__":
    main()
