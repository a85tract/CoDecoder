"""winmark.py MARKS pipeline <sv_speed pipeline arguments>: runs `sv_speed pipeline` of the tree on PYTHONPATH,
unchanged, and appends to MARKS one JSON line with the exact Unix-time bounds of the pipeline's timed part, so that
clockflags.py can take a timing's clock samples over exactly that window (the clean repeat of RESULTS.md section 3.1).
Lives outside the deployed tree, so the code digest the pipeline compares with the lock's stays the lock's.

Python 3.12's sys.monitoring reports each line of sv_speed.pipeline once (the callback returns DISABLE, which removes
the event at that line), so the timed loop runs uninstrumented after its first pass. At three lines the callback reads
the timer values the pipeline took itself (the frame's locals):
  single: t0 of shot 0 (first pass of `lat[i] = ...`, which runs from i = 0) to t1 of the last shot (the line after
    the loop), so the window includes the work between shots that the latencies leave out;
  batched: t0 (set just before the loop) to t0 + wall (the line after `wall = ...`).
A perf_counter value x becomes Unix time as u - (q - x), with (u, q) = (time.time(), perf_counter()) read together."""
import json
import os
import platform
import sys
import time


def main():
    marks, args = sys.argv[1], sys.argv[2:]
    sys.argv = ["sv_speed"] + args
    from surface import sv_speed
    code = sv_speed.pipeline.__code__
    src = [x.strip() for x in open(code.co_filename)]

    def find(text, start=0):
        hits = [k + 1 for k in range(start, len(src)) if src[k] == text]
        if not hits:
            raise SystemExit(f"winmark: no line {text!r} in {code.co_filename}")
        return hits

    a = find("lat[i] = 1e6 * (t1 - t0)")
    w = find("wall = time.perf_counter() - t0")
    if len(a) != 1 or len(w) != 1 or src[w[0]] != "if pool is not None:":
        raise SystemExit(f"winmark: {code.co_filename} is not the pipeline this wrapper reads")
    want = {a[0]: "first", find("fail = pred != obs", a[0])[0]: "last", w[0] + 1: "wall"}
    M = sys.monitoring
    tool = next(k for k in range(6) if M.get_tool(k) is None)
    M.use_tool_id(tool, "winmark")
    got = {}

    def line(_, ln):
        if ln in want:
            try:
                u, q = time.time(), time.perf_counter()
                v = sys._getframe(1).f_locals
                got[want[ln]] = {k: v.get(k) for k in ("i", "t0", "t1", "wall")} | {"u": u, "q": q}
            except Exception as e:             # never into the pipeline
                got[want[ln]] = {"error": repr(e)}
        return M.DISABLE

    M.register_callback(tool, M.events.LINE, line)
    M.set_local_events(tool, code, M.events.LINE)
    out = {"pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID", ""), "host": platform.node(), "args": args}
    try:
        sv_speed.main()
    finally:
        try:
            unix = lambda g, x: g["u"] - (g["q"] - x)
            if "wall" in got:
                g = got["wall"]
                out |= {"mode": "batched", "start": unix(g, g["t0"]), "end": unix(g, g["t0"] + g["wall"])}
            elif "first" in got and "last" in got:
                f, g = got["first"], got["last"]
                out |= {"mode": "single", "start": unix(f, f["t0"]), "end": unix(g, g["t1"]),
                        "first_i": f["i"], "last_i": g["i"]}
            if "start" in out:
                out["secs"] = out["end"] - out["start"]
                for k in ("start", "end"):
                    out[k + "_local"] = time.strftime("%H:%M:%S", time.localtime(out[k])) + f"{out[k] % 1:.6f}"[1:]
        except Exception as e:
            out["error"] = repr(e)
        out["raw"] = got
        with open(marks, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
