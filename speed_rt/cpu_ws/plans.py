"""SPEED-RT CPU half on the Windows CPU host: the plans of timed windows that speed_rt/cpu_ws/harness.py runs.

A job is one fresh single-threaded process (one repetition of one CPU part). The plan packs the jobs of each repetition
into windows of at most 8 (rep-major: every job of repetition 1, then of repetition 2, ...). Inside a repetition the
jobs are sorted by their expected length, longest first, so that a window's jobs end at about the same time (the
harness keeps every slot busy with a filler until the window's last job ends). Inside a window the two sides of a
comparison alternate across the slots (ours, baseline, ours, ...), and the order is reversed in every other window, so
neither side keeps a slot or a position. The parts:
  surf_base   plain and correlated PM per block at d = 7-17 (timing.cpu_term ref [--plain], timing chunk 0)
  surf_term   our pairs' CPU terms: correlated PM on the routed shots of each network (timing.cpu_term term; the
              mask from the hand-off, {handoff}/surface/<role>/val_<network>.npz)
  surf_share  Ising's PyMatching shares: plain and correlated PM on the residual detectors of Ising M1 and M4's counted
              plans, d = 7-17 (timing.cpu_term share; {handoff}/ising/<role>/res_d<d>_<M>_val000.npz)
  gref        the Grace fit's 18 reference points (speed_rt.gref time): plain PM d = 13, 21, 31 and correlated PM
              d = 5, 9, 13, 17, 21, 31, each at p = 0.003 and 0.006
  gs_fe       Decoder Switching's front end (speed_rt.gapsw.gs_time fe), d = 7-17
  gs_term     Decoder Switching's routed correlated PM (speed_rt.gapsw.gs_time term), d = 7-17
  bb          BB72 and gross: the baselines (sector Relay-BP, full-DEM Relay-BP, ldpc BP-OSD) and our Relay-BP terms
              (speed_rt/bb/srt_bb.py; the job list is bb_jobs.json beside this file)
Placeholders the harness fills: {py} {core} {out} {root} and the config's vars ({code} {data} {relay} {handoff}
{sessionref_us}). Roles: the hand-off of d = 7, 9, 11 comes from role "low", of d = 13, 15, 17 from "high".
Each job names the hand-off units it needs ("needs"); the harness starts a window once they are in the tmpfs. A
comparison's two sides need the same units, so they stay interleaved: surf_base and surf_term need "surface",
surf_share "ising", BB's baselines "bb/draw" and our BB terms "bb"; gref and Decoder Switching need none. Decoder
Switching's processes run in windows of their own (gs_time's own check rejects the window it runs in).

    python plans.py --parts gref,gs_fe,gs_term,surf_base,surf_term,surf_share,bb --reps 3 --out plan.json [--dry]
    python plans.py --show plan.json                 (windows, jobs and the expected length of each window)
--dry shrinks the long parts (gref --limit 2000 at d >= 21, smaller --first for BB, --chunks 4 for Decoder Switching); a dry plan
never names a timing set: the config's {data} points at the dry draws."""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SURF_ENV = {"PYTHONPATH": "{code}/surface_v2:{code}", "PYTHONDONTWRITEBYTECODE": "1"}
DS = (7, 9, 11, 13, 15, 17)
# the networks whose CPU term is timed (d, mask name); three SURF-v2 networks route nothing (tau = 0) and are kept,
# as the draft times them
NETS = [(7, "d7_b1w64_sv2"), (7, "d7_b1w64_r740"), (9, "d9_b3_sv2s1"), (9, "d9_b3_r764"), (11, "d11_b5_sv2"),
        (11, "d11_b5_r777"), (13, "d13_b5_sv2"), (13, "d13_b5_r775"), (15, "d15_b5_sv2"), (15, "d15_b5_r782"),
        (17, "d17_b5_sv2"), (17, "d17_b5_r787_q08"), (17, "d17_b5_r787_q10")]
GREF = [("plain", 13), ("plain", 21), ("plain", 31), ("corr", 5), ("corr", 9), ("corr", 13), ("corr", 17),
        ("corr", 21), ("corr", 31)]
# expected seconds per process on one Xeon Gold 6326 vCPU (start-up, build and decode), from the draft's records;
# they only order the jobs, the dry run replaces them (est_dry.json)
EST = {("surf_base", "plain"): {7: 6, 9: 7, 11: 9, 13: 11, 15: 15, 17: 19},
       ("surf_base", "corr"): {7: 7, 9: 9, 11: 12, 13: 17, 15: 25, 17: 33},
       "surf_term": {7: 15, 9: 15, 11: 20, 13: 25, 15: 45, 17: 70},
       ("gref", "plain"): {(13, 3): 6, (13, 6): 7, (21, 3): 11, (21, 6): 16, (31, 3): 35, (31, 6): 60},
       ("gref", "corr"): {(5, 3): 3, (5, 6): 3, (9, 3): 4, (9, 6): 5, (13, 3): 7, (13, 6): 9, (17, 3): 13,
                          (17, 6): 19, (21, 3): 28, (21, 6): 45, (31, 3): 80, (31, 6): 130},
       "gs_fe": {7: 6, 9: 13, 11: 24, 13: 48, 15: 91, 17: 168},           # 50,000 shots per d
       "gs_term": {7: 3, 9: 3, 11: 4, 13: 5, 15: 8, 17: 19}}


def est_override():
    p = os.path.join(HERE, "est_dry.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def role(d):
    return "low" if d <= 11 else "high"


def surf_base(rep, dry):
    for dec in ("plain", "corr"):
        for d in DS:
            argv = ["{py}", "-m", "timing.cpu_term", "ref", "--d", str(d), "--val", "{data}/surf/view", "--rep",
                    str(rep), "--core", "{core}", "--out", "{out}"] + (["--plain"] if dec == "plain" else [])
            yield dict(id=f"base_{dec}_d{d}_r{rep}", part="surf_base", side="baseline", cwd="{code}", env=SURF_ENV,
                       argv=argv, est_s=EST[("surf_base", dec)][d], needs=["surface"])


def surf_term(rep, dry):
    for d, net in NETS:
        argv = ["{py}", "-m", "timing.cpu_term", "term", "--d", str(d), "--val", "{data}/surf/view", "--mask",
                f"{{handoff}}/surface/{role(d)}/val_{net}.npz", "--rep", str(rep), "--core", "{core}", "--out", "{out}"]
        yield dict(id=f"term_{net}_r{rep}", part="surf_term", side="ours", cwd="{code}", env=SURF_ENV, argv=argv,
                   est_s=EST["surf_term"][d], needs=["surface"])


def surf_share(rep, dry):
    for dec in ("plain", "corr"):
        for m in ("M1", "M4"):
            for d in DS:
                argv = ["{py}", "-m", "timing.cpu_term", "share", "--d", str(d), "--model", m, "--res",
                        f"{{handoff}}/ising/{role(d)}", "--rep", str(rep), "--core", "{core}", "--out", "{out}"] + \
                       (["--plain"] if dec == "plain" else [])
                yield dict(id=f"share_{dec}_{m}_d{d}_r{rep}", part="surf_share", side="baseline", cwd="{code}",
                           env=SURF_ENV, argv=argv, est_s=EST[("surf_base", dec)][d], needs=["ising"])


def gref(rep, dry):
    for dec, d in GREF:
        for p in ("0.003", "0.006"):
            pt = p[-1]
            argv = ["{py}", "-m", "speed_rt.gref", "time", "--dec", dec, "--d", str(d), "--p", p, "--shots",
                    "{data}/ref/shots", "--manifest", "{data}/ref/srt-ref.json", "--rep", str(rep), "--core", "{core}",
                    "--out", "{out}"] + (["--limit", "2000"] if dry and d >= 21 else [])
            yield dict(id=f"gref_{dec}_d{d}_p{pt}_r{rep}", part="gref", side="reference", cwd="{code}", env=SURF_ENV,
                       argv=argv, est_s=EST[("gref", dec)][(d, int(pt))])


def gs(kind):
    def gen(rep, dry):
        for d in DS:
            argv = ["{py}", "-m", "speed_rt.gapsw.gs_time", kind, "--d", str(d), "--masks", "{data}/gs/masks",
                    "--shots", "{data}/gs/shots", "--manifest", "{data}/gs/srt-gs.json", "--core", "{core}", "--rep",
                    str(rep), "--probe-file", "{data}/probe/d9_p0.006_X_val_000.npz", "--probe-ref",
                    "{sessionref_us}", "--out", "{out}"] + (["--chunks", "4"] if dry else [])
            yield dict(id=f"gs_{kind}_d{d}_r{rep}", part=f"gs_{kind}", side="opponent", cwd="{code}",
                       env={"PYTHONPATH": "{code}:{code}/surface_v2", "PYTHONDONTWRITEBYTECODE": "1"}, argv=argv,
                       est_s=EST[f"gs_{kind}"][d] / (5 if dry else 1))
    return gen


def bb_jobs(rep, dry, warm):
    for j in json.load(open(os.path.join(HERE, "bb_jobs.json"))):
        if (j.get("kind") == "warmup") != warm:
            continue
        argv = list(j["argv"])
        if dry:
            for k, v in (j.get("dry_argv_overrides") or {}).items():
                argv[argv.index(k) + 1] = str(v)
        ra = j.get("rep_arg", "--proc")
        if ra in argv:
            argv[argv.index(ra) + 1] = str(rep)
        else:
            argv += [ra, str(rep)]
        yield dict(id=f"{j['id']}_r{rep}", part=j["part"], side=j.get("side", "baseline"), cwd=j["cwd"],
                   env=j.get("env", {}), argv=argv, est_s=j.get("est_s", 60) / (10 if dry else 1),
                   needs=j.get("needs", ["bb" if j.get("needs_preds") else "bb/draw"]))


def bb(rep, dry):
    return bb_jobs(rep, dry, False)


def bb_warm(rep, dry):
    """The BB warm-up processes (untimed, rep 0 in their CSV): run once, before the first window of a session."""
    return bb_jobs(0, dry, True) if rep == 1 else iter(())


PARTS = dict(surf_base=surf_base, surf_term=surf_term, surf_share=surf_share, gref=gref, gs_fe=gs("fe"), gs_term=gs("term"), bb=bb,
             bb_warm=bb_warm)


def pack(jobs, slots=8):
    """Windows of at most `slots` jobs, longest first; sides alternate inside a window, reversed every other window."""
    jobs = sorted(jobs, key=lambda j: (-j["est_s"], j["id"]))
    wins = [jobs[i:i + slots] for i in range(0, len(jobs), slots)]
    out = []
    for w, js in enumerate(wins):
        ours = [j for j in js if j["side"] == "ours"]
        rest = [j for j in js if j["side"] != "ours"]
        order = []
        while ours or rest:
            for src in ((ours, rest) if w % 2 == 0 else (rest, ours)):
                if src:
                    order.append(src.pop(0))
        if w % 2:
            order.reverse()
        out.append(order)
    return out


def needs_groups(jobs):
    """The jobs grouped by the hand-off units they need, the groups that need nothing first, so that a window never
    waits for a unit one of its jobs does not need. Decoder Switching's processes get windows of their own: gs_time's
    own host check (exit 4) rejects the window it runs in, and it should cost no other part an attempt."""
    g = {}
    for j in jobs:
        g.setdefault((len(j.get("needs", [])), tuple(sorted(j.get("needs", []))), j["part"].startswith("gs_")), []).append(j)
    return [g[k] for k in sorted(g)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", default="gref,gs_fe,gs_term,surf_base,surf_term,surf_share,bb")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--order", choices=("lpt", "rep"), default="rep",
                    help=("lpt: the jobs of all repetitions packed together, longest first (a long job's repetitions can "
                          "share a window); rep: repetition by repetition, every job of repetition 1 first"))
    ap.add_argument("--out", default="")
    ap.add_argument("--show", default="")
    ap.add_argument("--no-needs", action="store_true", help="no job waits for a hand-off (inputs already in place)")
    a = ap.parse_args()
    if a.show:
        plan = json.load(open(a.show))
        tot = 0
        for w in plan["windows"]:
            m = max(j["est_s"] for j in w["jobs"])
            tot += m
            print(w["id"], f"~{m:.0f} s", " ".join(f"{j['id']}({j['side'][0]})" for j in w["jobs"]))
        print(f"{len(plan['windows'])} windows, {sum(len(w['jobs']) for w in plan['windows'])} jobs, ~{tot / 60:.0f} min"
              " of timed windows (plus about 10 s of reference segments per window)")
        return
    ov = est_override()
    windows = []
    groups = [list(range(1, a.reps + 1))] if a.order == "lpt" else [[r] for r in range(1, a.reps + 1)]
    for reps in groups:
        jobs = [j for rep in reps for p in a.parts.split(",") for j in PARTS[p](rep, a.dry)]
        for j in jobs:
            j["est_s"] = ov.get(j["id"].rsplit("_r", 1)[0], j["est_s"])
        warm = [j for j in jobs if j["id"].endswith("_r0")]                # warm-ups first, in windows of their own
        jobs = [j for j in jobs if not j["id"].endswith("_r0")]
        if a.no_needs:
            for j in warm + jobs:
                j["needs"] = []
        for js in pack(warm, a.slots) + [x for g in needs_groups(jobs) for x in pack(g, a.slots)]:
            windows.append(dict(id=f"w{len(windows) + 1:03d}", rep=reps[0] if len(reps) == 1 else None, jobs=js,
                                needs=sorted({n for j in js for n in j.get("needs", [])})))
    plan = dict(what="SPEED-RT CPU plan (speed_rt/cpu_ws/plans.py)", parts=a.parts, reps=a.reps, dry=a.dry, order=a.order,
                slots=a.slots, windows=windows)
    with open(a.out, "x") as fh:
        json.dump(plan, fh, indent=1)
    print(f"{a.out}: {len(windows)} windows, {sum(len(w['jobs']) for w in windows)} jobs")


if __name__ == "__main__":
    main()
