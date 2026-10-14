"""The surface stage's quantities (the stage's section, "Analysis"): from one pod role's session records to every quantity of
its distances, by the frozen statistics. A role is one pod: S-low (d = 7, 9, 11) or S-high (d = 13, 15, 17), its GPU items
G1-G6 and its CPU parts C-ref, C-base, C-term, C-share from one session (speed_rt/pod_s.py).
  ours        a = the slowest of the five accepted GPU-term processes; c = the d = 7 kernel's handoff (the largest of
              five) or max(1.1 us, the router's estimate) at d = 9-17; b_xeon = the slowest of the three CPU-term
              repetitions; composite = a + b + c, b on the basis named
  references  the PM baselines per block (C-base) and the PM shares after Ising (C-share), timed by the CPU worker: the FASTEST of
              three fresh processes (stage 1's rule: a baseline takes its fastest); the Grace fit's 18 points (C-ref): the SLOWEST of
              three (the smaller factor: it goes against the core claim)
  Ising       the counted plan's time (speed_rt/ising_select.py: the fastest accepted repetition of the fastest
              validated plan) plus its PM share on the basis named
  BP-OSD      the fastest of three accepted repetitions (CUDA-Q on the pod, its host CPU), not converted
Bases (a Grace-basis estimate, no Grace hardware):
  grace       every CPU time x the factor of its decoder, fitted on this pod's references (speed_rt/grace_fit.py): plain-PM
              terms (the PM baseline, the PM shares after Ising) take the applied plain factor, correlated-PM terms (our
              term b, the correlated-PM baseline, the correlated shares after Ising) the applied correlated factor
  xeon        the CPU worker's own times, unconverted (D4: the measured-CPU disclosure)
A quantity that is not established is None with its reason; every comparison that needs it is withheld. A session that does
not qualify (no DONE.json, or a pre-screen that did not pass) establishes nothing; the CPU quantities come from the CPU worker's session
(the Linux workstation: cpu_session.json, cpu_attempts.json, cpu/) and are judged by its own end record."""
import glob
import hashlib
import json
import os
import re

from speed_rt import grace_fit

DS = (7, 9, 11, 13, 15, 17)
ROLES = {"low": (7, 9, 11), "high": (13, 15, 17)}
MODELS = ("M1", "M4")
ARMS = ("minllr", "bp60")
Q10 = "d17_b5_r787_q10"
Q08 = "d17_b5_r787_q08"
REPS = (1, 2, 3)


def load(p):
    with open(p) as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


class Missing(Exception):
    pass


def need(cond, why):
    if not cond:
        raise Missing(why)


def slots(att):
    """{(item, key, round): (accepted attempt or None, every attempt)} from a session's attempts.json."""
    out = {}
    for x in att:
        k = (x["item"], x["key"], x["round"])
        acc, every = out.get(k, (None, []))
        every.append(x)
        out[k] = (acc or (x if x["accepted"] else None), every)
    return out


def established(sl, item, key, rounds):
    """The accepted records of an item's slots, or Missing naming the first slot that is not established."""
    recs = []
    for r in rounds:
        acc = sl.get((item, key, r), (None, []))[0]
        need(acc is not None, f"{item} {key}: round {r} not established")
        recs.append(acc)
    return recs


def rounds_of(sl, item, dry):
    """The rounds an item is established in: the registered three; dry runs: the repetitions the session ran."""
    if not dry:
        return REPS
    r = sorted({k[2] for k in sl if k[0] == item})
    return tuple(r) if r else REPS


def gpu_rounds(sl, item, default, dry):
    """The registered rounds of a GPU item; dry runs: the rounds the session ran."""
    if not dry:
        return default
    r = sorted({k[2] for k in sl if k[0] == item})
    return r or default


def qualifies(S, dry):
    """(True, None) when the GPU session reached its end (DONE.json) with its pre-screen record passed, else (False, the reason)."""
    if not os.path.exists(os.path.join(S, "attempts.json")):
        return False, "no session records"
    if not os.path.exists(os.path.join(S, "DONE.json")):
        return False, "the session did not reach its end (no DONE.json)"
    if not dry:                                   # a registered session: a committed tree, a passed pre-screen, not a dry run
        meta = load(os.path.join(S, "META.json"))
        if meta.get("dry") or load(os.path.join(S, "DONE.json")).get("dry"):
            return False, "a dry session"
        if not re.fullmatch(r"[0-9a-f]{40}", str(meta.get("deployed") or "")):
            return False, "the session did not run from a committed tree (META.json deployed)"
        pre = os.path.join(S, "prescreen.json")
        if not os.path.exists(pre) or not load(pre).get("passed"):
            return False, "the pod's pre-screen record is missing or did not pass"
        bi = os.path.join(S, "BUILD_INPUTS.json")         # the plan builds' record (speed_rt/build_inputs.py): the builds may run
        if not os.path.exists(bi):                         # before F_2, so every tree file they read must equal this tree's (F_2's)
            return False, "no record of the plan builds (BUILD_INPUTS.json)"
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bad = [f for f, h in sorted(load(bi)["tree"].items())
               if not os.path.isfile(os.path.join(root, f)) or sha256_file(os.path.join(root, f)) != h]
        if bad:
            return False, f"the plans were built from files other than F_2's ({len(bad)}: {', '.join(bad[:3])})"
    return True, None


def qualifies_cpu(C, dry):
    """(True, None) when the CPU worker's session (C = RECORDS/cpu: cpu_session.json, cpu_attempts.json, cpu/) reached its end, else
    (False, the reason). The CPU worker's own acceptance of each window is its attempts' accepted flag."""
    f = os.path.join(C, "cpu_session.json")
    if not os.path.exists(f) or not os.path.exists(os.path.join(C, "cpu_attempts.json")):
        return False, "no CPU session records (the Linux workstation)"
    m = load(f)
    if not m.get("done"):
        return False, "the CPU session did not reach its end (cpu_session.json: done)"
    if not dry and m.get("dry"):
        return False, "a dry CPU session"
    return True, None


def gpu_quantities(role, S, masks, freeze, cfg, shots_digest, dry=False):
    """The GPU-side quantities of a role (S: its session directory, masks: its routing masks), with the checks of the
    freeze: every network's mask records the timing set's digest (the draw manifest's shots_sha256), which every timed
    process checked against the shots it read (timing.deployed.load). dry: the dry runs' --limit is allowed."""
    import numpy as np
    q, why, reps = {}, {}, {}
    att = load(os.path.join(S, "attempts.json"))
    sl = slots(att)
    ds = ROLES[role]
    nets = {k: n for k, n in freeze["surface"]["networks"].items() if n["d"] in ds}
    code = freeze["surface"]["code_digest"]
    for name, n in sorted(nets.items()):
        d = n["d"]
        mk = os.path.join(masks, f"val_{name}.npz")
        if not os.path.exists(mk) or str(np.load(mk)["shots_sha256"]) != shots_digest["shots_sha256"][str(d)]:
            for kind in ("a", "c"):
                q[f"S.{name}.{kind}"], why[f"S.{name}.{kind}"] = None, f"{name}: the mask is missing or of other shots"
            continue
        for kind, item, rounds in (("a", "G1", range(1, 6)), ("c", "G2", range(1, 6 if d == 7 else 2))):
            key = f"S.{name}.{kind}"
            try:
                recs = established(sl, item, name, gpu_rounds(sl, item, rounds, dry))
                vals = []
                for x in recs:
                    r = load(os.path.join(S, x["record"]))
                    need(r["accepted"] and r["network"] == name and r["network_sha256"] == n["sha256"], f"{x['record']}")
                    need(r["code_digest"] == code and (dry or r.get("limit") is None), f"{x['record']}: code or limit")
                    need(r["gpu"] == cfg["hosts"]["pod_gpu"], f"{x['record']}: GPU {r['gpu']}")
                    if kind == "c" and d != 7:
                        need(r["tau"] == n["tau"], f"{x['record']}: tau")
                        q[f"S.{name}.handoff_routed"] = r["routed"]
                        q[f"S.{name}.complete_path_us"] = r["complete_path"]["mean"]
                    vals.append(r["handoff_us"] if kind == "c" else r["per_shot_mean"])
                q[key] = max(vals)
                reps[key] = vals
            except (Missing, KeyError, OSError) as e:
                q[key], why[key] = None, str(e)
    if role == "low":                                                  # G4: the Triton engine at d = 7
        try:
            recs = established(sl, "G4", "triton7_d7", gpu_rounds(sl, "G4", range(1, 6), dry))
            vals = []
            for x in recs:
                r = load(os.path.join(S, x["record"]))
                need(r["accepted"] and r["code_digest"] == code and (dry or r.get("limit") is None), x["record"])
                vals.append(r["per_shot_mean"])
            q["S.triton7.a"], reps["S.triton7.a"] = max(vals), vals
        except (Missing, KeyError, OSError) as e:
            q["S.triton7.a"], why["S.triton7.a"] = None, str(e)
    for d in ds:                                                       # G6: BP-OSD, fastest of three accepted repetitions
        for arm in ARMS:
            key = f"S.bposd.{arm}.{d}"
            try:
                recs = established(sl, "G6", f"d{d}_{arm}", gpu_rounds(sl, "G6", range(1, 4), dry))
                vals = []
                for x in recs:
                    r = load(os.path.join(S, x["record"]))
                    need(r["accepted"] and (dry or r["shots"] == cfg["items"]["G6"]["shots"]), x["record"])
                    need(dry or r["shots_sha256"] == shots_digest["chunk0_prefix"]["sha256"][str(d)], f"{x['record']}: shots")
                    vals.append(r["mean_us"])
                q[key], reps[key] = min(vals), vals
            except (Missing, KeyError, OSError) as e:
                q[key], why[key] = None, str(e)
    from speed_rt import ising_select                                  # G5: the counted Ising plan per model and d
    sel = ising_select.select(S, ds)
    for m in MODELS:
        for d in ds:
            key = f"S.ising.{m}.{d}"
            g = [x for x in sl if x[0] == "G5-graph" and x[1].startswith(f"d{d}_")]
            if (m, d) in sel and g and all(sl[x][0] is not None for x in g):
                q[key] = sel[(m, d)]["value_us"]
                q[f"S.ising.{m}.{d}.plan"] = sel[(m, d)]["plan"]
                q[f"S.ising.{m}.{d}.source"] = sel[(m, d)]["source"]
            else:
                q[key], why[key] = None, f"no accepted graph processes or no validated plan of {m} at d = {d}"
    return q, why, reps


def ref_key(role, dec, d, p):
    return f"N.ref.{role}.{dec}.d{d}.p{round(p * 1000)}"


def cpu_reader(C, freeze, dry=False):
    """(get, meta, code): get(item, key) = the accepted records of a CPU slot's three repetitions (Missing when one is not established
    or is not on the CPU session's CPU, pinned, or a --limit run), from the CPU worker's cpu_attempts.json and records."""
    att = load(os.path.join(C, "cpu_attempts.json"))
    sl = slots(att)
    meta = load(os.path.join(C, "cpu_session.json"))

    def get(item, key):
        recs = []
        for x in established(sl, item, key, rounds_of(sl, item, dry)):
            r = load(os.path.join(C, x["record"]))
            need(r.get("cpu") == meta["cpu"] and r.get("pinned") is True,
                 f"{x['record']}: not the CPU session's CPU, or not pinned")
            need(r.get("limit") is None or dry, f"{x['record']}: --limit")
            r["_path"] = x["record"]
            recs.append(r)
        need(dry or len(recs) == 3, f"{item} {key}: {len(recs)} repetitions")
        return recs
    return get, meta, freeze["surface"]["code_digest"]


def ref_points(C, freeze, ref_manifest, circuits, dry=False):
    """The Grace fit's 18 reference points from the CPU worker's C-ref records alone (no pod record is read): {(decoder, d, p): dict(us =
    the SLOWEST of the three repetitions' per-shot times, reps = the three values, records = their paths)} and {(decoder, d, p): the
    reason} for a point that is not established. Every record names its circuit and chunk, which must be the registered ones."""
    get, meta, code = cpu_reader(C, freeze, dry)
    pts, why = {}, {}
    for dec, d, p in grace_fit.points():
        try:
            v, paths = [], []
            for r in get("C-ref", f"{dec}_d{d}_p{round(p * 1000)}"):
                s = f"srt-ref-d{d}-p{round(p * 1000)}"
                need(r["decoder"] == dec and r["d"] == d and r["p"] == p and r["code_digest"] == code, f"{s}: record")
                need(r["circuit_sha256"] == circuits[s]["sha256"], f"{s}: circuit")
                need(r["shots_file_sha256"] == ref_manifest["files"][r["shots_file"]]["sha256"], f"{s}: shots")
                need(dry or r["shots"] == 10000, f"{s}: {r['shots']} shots")
                v.append(r["per_shot_us"])
                paths.append(r["_path"])
            pts[(dec, d, p)] = dict(us=max(v), reps=v, records=paths)          # the Grace references: the slowest of three
        except (Missing, KeyError, OSError) as e:
            pts[(dec, d, p)], why[(dec, d, p)] = None, str(e)
    return pts, why


def fit_s2(C, freeze, ref_manifest, circuits, dry=False):
    """The stage's ONE Grace fit (fit_s2.json): grace_fit.fit over the 18 points of ref_points, whole output plus every point's raw
    repetitions and record digests. complete False (no fit) when a point is not established: `missing` names them. Deterministic."""
    get, meta, code = cpu_reader(C, freeze, dry)
    pts, why = ref_points(C, freeze, ref_manifest, circuits, dry)
    raw = {}
    for (dec, d, p), v in sorted(pts.items()):
        key = f"{dec}_d{d}_p{round(p * 1000)}"
        raw[key] = None if v is None else dict(decoder=dec, d=d, p=p, xeon_us_per_shot=v["us"], statistic="slowest of three",
                                               repetitions_us_per_shot=v["reps"], records=v["records"],
                                               records_sha256=[sha256_file(os.path.join(C, r)) for r in v["records"]])
    ok = not why
    return dict(what="SPEED-RT stage 2's Grace fit (one fit for every family; speed_rt/analysis_s.fit_s2, grace_fit.fit)"
                     + (" - DRY: never a result" if dry else ""), dry=dry, complete=ok, cpu=meta["cpu"], code_digest=code,
                statistic="the slowest of three fresh processes per point (STAGE2_common C3)",
                missing={f"{k[0]}_d{k[1]}_p{round(k[2] * 1000)}": w for k, w in sorted(why.items())},
                raw_points=raw, fit=grace_fit.fit({k: v["us"] for k, v in pts.items()}) if ok else None)


def cpu_quantities(role, C, res_dir, freeze, session_q, shots_digest, ref_manifest, circuits, dry=False):
    """The CPU quantities of a role from the CPU worker's session (the Linux workstation) accepted C-slots, by stage 1's statistics: the Grace fit's
    reference points (the SLOWEST of three; every record names its circuit and chunk, which must be the registered ones), the PM
    baselines per block (the FASTEST of three), our CPU terms (the SLOWEST of three) and the shares after Ising (the FASTEST of three,
    timed on the counted plan's residuals; the baseline is Ising's plan time plus this share)."""
    q, why, reps = {}, {}, {}
    ds = ROLES[role]
    get, meta, code = cpu_reader(C, freeze, dry)
    pts, pwhy = ref_points(C, freeze, ref_manifest, circuits, dry)
    for (dec, d, p), v in pts.items():                                    # C-ref: the fit's 18 points
        key = ref_key(role, dec, d, p)
        if v is None:
            q[key], why[key] = None, pwhy[(dec, d, p)]
        else:
            q[key], reps[key] = v["us"], v["reps"]
    for d in ds:                                                          # C-base: the PM baselines per block
        for kind, label in (("corr", "ref_corr"), ("plain", "ref_plain")):
            key = f"N.{label}.{d}"
            try:
                v = []
                for r in get("C-base", f"{kind}_d{d}"):
                    need(r["what"] in ("ref", "ref_plain") and r["d"] == d and r["code_digest"] == code and
                         r["kind"] == ("plain" if kind == "plain" else "correlated"), f"{key}: record")
                    v.append(r["per_block_us"])
                q[key], reps[key] = min(v), v                             # a baseline: the fastest of three
            except (Missing, KeyError, OSError) as e:
                q[key], why[key] = None, str(e)
    nets = {k: n for k, n in freeze["surface"]["networks"].items() if n["d"] in ds}
    todo = sorted(nets.items()) + ([(Q10, dict(d=17))] if role == "high" else [])
    for name, n in todo:                                                  # C-term: our CPU term b
        d = n["d"]
        key = f"N.term.{name}"
        try:
            v = get("C-term", name)
            need(len({(r["routed"], r["mask_sha256"]) for r in v}) == 1 and
                 all(r["network"] == name and r["shots_sha256"] == shots_digest["shots_sha256"][str(d)] and
                     r["code_digest"] == code for r in v), f"{key}: the routed shots or the mask differ")
            hr = session_q.get(f"S.{name}.handoff_routed")
            need(dry or d == 7 or name == Q10 or hr is None or v[0]["routed"] == hr,
                 f"{key}: the routed shots differ from the session's handoff")
            slow = max(v, key=lambda r: r["term_us"])
            q[key], reps[key] = slow["term_us"], [r["term_us"] for r in v]
            q[f"N.cond.{name}"] = slow["conditional_mean_us"]
            q[f"N.routed_fraction.{name}"] = slow["routed_fraction"]
        except (Missing, KeyError, OSError) as e:
            q[key], why[key] = None, str(e)
    for m in MODELS:                                                      # C-share: the shares after Ising
        for d in ds:
            f = os.path.join(res_dir, f"res_d{d}_{m}_val000.json")
            plan = session_q.get(f"S.ising.{m}.{d}.plan")
            rj = load(f) if os.path.exists(f) else {}
            res = rj.get("res_content_sha256") if plan is not None and rj.get("plan") == plan else None
            for kind, label in (("corr", "share_corr"), ("plain", "share_plain")):
                key = f"N.{label}.{m}.{d}"
                try:
                    need(res is not None, f"no residuals of the counted plan of {m} at d = {d}")
                    v = []
                    for r in get("C-share", f"{kind}_{m}_d{d}"):
                        need(r["what"] == "share" and r["d"] == d and r["model"] == m and r["code_digest"] == code and
                             r["residuals_content_sha256"] == res, f"{key}: not the counted plan's residuals")
                        v.append(r["per_block_us"])
                    q[key], reps[key] = min(v), v                         # a baseline's CPU share: the fastest of three
                except (Missing, KeyError, OSError) as e:
                    q[key], why[key] = None, str(e)
    return q, why, reps


def fit_role(role, N, why):
    """The pod's Grace fit (speed_rt/grace_fit.fit) from its 18 reference points, or None with the reason."""
    pts = {k: N.get(ref_key(role, *k)) for k in grace_fit.points()}
    miss = [f"{k[0]} d{k[1]} p{round(k[2] * 1000)}" for k, v in pts.items() if v is None]
    if miss:
        why[f"FIT.{role}"] = f"{len(miss)} reference points not established: {', '.join(miss[:6])}"
        return None
    return grace_fit.fit(pts)


def conv(v, kind, fit):
    return None if v is None or fit is None else grace_fit.to_grace(v, kind, fit)


def bases(N, fit, ds):
    """{basis: {quantity: value}} of a role's baselines' CPU values on the grace (the pod's fitted factors) and xeon
    (unconverted) bases, and each basis' factor for our term b."""
    g, x = {}, {}
    for d in ds:
        for key, kind, label in (("N.ref_plain", "plain", "pm"), ("N.ref_corr", "corr", "corr_pm")):
            x[f"{label}.{d}"] = N.get(f"{key}.{d}")
            g[f"{label}.{d}"] = conv(N.get(f"{key}.{d}"), kind, fit)
        for m in MODELS:
            for key, kind, label in (("N.share_plain", "plain", "share"), ("N.share_corr", "corr", "corrshare")):
                x[f"{label}.{m}.{d}"] = N.get(f"{key}.{m}.{d}")
                g[f"{label}.{m}.{d}"] = conv(N.get(f"{key}.{m}.{d}"), kind, fit)
    g["term_factor"] = None if fit is None else fit["applied"]["corr"]
    x["term_factor"] = 1.0
    return dict(grace=g, xeon=x)


def composites(Sq, N, B, freeze, ds):
    """{basis: {network: composite}} and the parts of a role's networks; b = b_xeon x the basis' factor for correlated PM."""
    out, parts = {}, {}
    nets = sorted((k, n) for k, n in freeze["surface"]["networks"].items() if n["d"] in ds)
    if 17 in ds:
        nets += [(Q10, dict(d=17))]
    for basis, base in B.items():
        out[basis] = {}
        for name, n in nets:
            a = Sq.get(f"S.{name}.a") if name != Q10 else Sq.get(f"S.{Q08}.a")
            c = Sq.get(f"S.{name}.c") if name != Q10 else Sq.get(f"S.{Q08}.c")
            bn, fac = N.get(f"N.term.{name}"), base["term_factor"]
            b = None if bn is None or fac is None else bn * fac
            out[basis][name] = None if None in (a, b, c) else a + b + c
            parts[f"{basis}.{name}"] = dict(a=a, b=b, c=c, b_xeon=bn, ratio=None if b is None or not bn else b / bn)
    return out, parts


def baselines(Sq, B, ds):
    """{basis: {(baseline, d): time}} for the eight surface baselines of a role's distances."""
    out = {}
    for basis, base in B.items():
        t = {}
        for d in ds:
            t[("pm", d)] = base.get(f"pm.{d}")
            t[("corr_pm", d)] = base.get(f"corr_pm.{d}")
            for m in MODELS:
                it = Sq.get(f"S.ising.{m}.{d}")
                for kind, label in (("share", f"ising_{m}_pm"), ("corrshare", f"ising_{m}_corr_pm")):
                    s = base.get(f"{kind}.{m}.{d}")
                    t[(label, d)] = None if it is None or s is None else it + s
            for arm in ARMS:
                t[(f"bposd_{arm}", d)] = Sq.get(f"S.bposd.{arm}.{d}")
        out[basis] = t
    return out


def role(role_name, R, C, freeze, cfg, manifest_surf, manifest_ref, circuits, dry=False):
    """Everything of one pod role from its records directory R (session/, masks/, res/) and the CPU worker's records C (cpu_session.json,
    cpu_attempts.json, cpu/): the quantities, the fit, the bases, the composites and baselines, and the reasons for what is missing. A GPU
    session that does not qualify has no GPU quantity; a CPU session that does not qualify has no CPU quantity (and no fit)."""
    S = os.path.join(R, "session")
    ds = ROLES[role_name]
    ok, reason = qualifies(S, dry)
    okc, reason_c = qualifies_cpu(C, dry)
    Sq, N, why, reps = {}, {}, {}, {}
    nets = [k for k, n in freeze["surface"]["networks"].items() if n["d"] in ds] + ([Q10] if role_name == "high" else [])
    if ok:
        Sq, why_s, reps_s = gpu_quantities(role_name, S, os.path.join(R, "masks"), freeze, cfg, manifest_surf, dry)
        why, reps = dict(why_s), dict(reps_s)
    else:
        why[f"SESSION.{role_name}"] = reason
        for name in nets:
            for k in ("a", "c"):
                Sq[f"S.{name}.{k}"], why[f"S.{name}.{k}"] = None, reason
        for d in ds:
            for arm in ARMS:
                Sq[f"S.bposd.{arm}.{d}"], why[f"S.bposd.{arm}.{d}"] = None, reason
            for m in MODELS:
                Sq[f"S.ising.{m}.{d}"], why[f"S.ising.{m}.{d}"] = None, reason
        if role_name == "low":
            Sq["S.triton7.a"], why["S.triton7.a"] = None, reason
    if okc:
        N, why_n, reps_n = cpu_quantities(role_name, C, os.path.join(R, "res"), freeze, Sq, manifest_surf, manifest_ref, circuits, dry)
        why, reps = dict(why, **why_n), dict(reps, **reps_n)
    else:
        why[f"CPU.{role_name}"] = reason_c
        for name in nets:
            N[f"N.term.{name}"], why[f"N.term.{name}"] = None, reason_c
        for d in ds:
            for key in ("ref_corr", "ref_plain"):
                N[f"N.{key}.{d}"], why[f"N.{key}.{d}"] = None, reason_c
    fit = fit_role(role_name, N, why) if okc else None
    if not okc:
        why[f"FIT.{role_name}"] = reason_c
    B = bases(N, fit, ds)
    comp, parts = composites(Sq, N, B, freeze, ds)
    base = baselines(Sq, B, ds)
    meta = load(os.path.join(S, "META.json")) if ok else None
    cmeta = load(os.path.join(C, "cpu_session.json")) if okc else None
    return dict(Sq=Sq, N=N, why=why, reps=reps, fit=fit, B=B, comp=comp, parts=parts, base=base, meta=meta, cpu_meta=cmeta,
                qualifies=ok, qualifies_cpu=okc)
