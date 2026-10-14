"""analysis_gs.py: the Decoder Switching stage's analysis (the stage's section: "Analysis"). Deterministic: no clock, no file
order, no randomness; run twice, the two outputs are byte-identical.

    python -m speed_rt.gapsw.analysis_gs --session SESSION --prescreen PRESCREEN --manifest srt-gs.json --masks MASKS
        --fit FIT.json --beacon speed_rt/gapsw/beacon.json --commits F B --out OUT
    python -m speed_rt.gapsw.analysis_gs --dry ... (no beacon check; --commits optional; never a result)

STATISTIC. Decoder Switching is the baseline of this comparison and there is no "our side" in this family (UPDATE 27, item 1: the
baseline takes the fastest of its three fresh processes): every term's value (front end, routed correlated PM) is the FASTEST of
its three repetitions' processes. The slowest and the median of three are sensitivity rows, deciding nothing. Every repetition's
value is kept beside it, raw and on the Grace basis. A term with a slot not established has no value (withheld).

GRACE BASIS. This family has no reference points of its own. The stage's ONE Grace fit is the surface family's: its 12 reference
sets srt-ref-* timed with speed_rt/gref.py and fitted by speed_rt/grace_fit.fit, written by the surface family's analysis as
FIT.json (grace_fit.fit's output plus the raw point times; its `applied` has the factors of the kinds plain, corr and other).
Decoder Switching is one of the "other" CPU decoders of the stage's rule: its front end and its routed shots' correlated PM are
both converted as kind "other", grace_fit.to_grace(us, "other", fit) (the common factor when the fit found one, otherwise the
geometric mean of the plain-PM and correlated-PM factors). The sensitivity rows convert the front end as kind "plain" and the
routed term as kind "corr" (each PM decoder's own factor), and the front end as kind "corr". FIT.json must re-derive: grace_fit.fit on its points' raw times gives its `applied`. `--fit none`:
the stage's fit was not established; no value is converted, every converted value is withheld, and no earlier value stands in.
Every raw Linux workstation time is kept beside its converted value, and every composite has its raw-basis twin (UPDATE 27, item 8).

It reads one session's records (SESSION/session.json, attempts.json, slots/<slot>/a<k>/rec.json), refuses what does not check
(the registered plan complete; every accepted process's host verdicts re-derived from its raw counters; the code, shots,
masks, policies, circuits and environment those of the session and the draw; every rejected attempt's reasons re-derived),
and writes:
  gs_numbers.json   every slot and attempt, the fit as read, every term, composite and descriptive row with its wording
  gs_fit.csv        the stage's fit as read from FIT.json: every reference point's raw time, NVIDIA's value, ratio, residual
  gs_times.csv      per d: the terms, raw and on the Grace basis (fastest, slowest, median of three), the composites
  gs_tables.md      the same as tables
A value whose slot was not established, or a conversion without the fit, is withheld (null in the JSON, "withheld"
in the tables), and no other value stands in for it. Records of older dry sessions may hold reference slots (kind "ref"):
a dry analysis does not read them."""
import argparse
import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from speed_rt import grace_fit  # noqa: E402
from speed_rt.gapsw import drive_gs, gslib as G, hostwin as H  # noqa: E402

REGISTERED_POLICY_SHA256 = {          # surf_v2/GAPSW_REGISTRATION.md
    7: "5219dc5aff5f80f6ee4655b789c86d55e2c3bc3c5be95941f731d67affb820d6",
    9: "cb3c8ff70c8ad0865f4e454f15fe0a7efb7b5c8b7f5f2d6e38c0944a3c3a9f47",
    11: "f9e1a874acfe6cb8c6b0b6ae3abfb2f011ecb109ce47913809b0e7f73ababff5",
    13: "f340ce5250090a1a64abce811bc090f0e733695dcea916883abf728aacc39571",
    15: "c14591d9e5fa387b64354f3904180400848f3ee9d6086e4e947bfcbb6ee2d27e",
    17: "922dd497ea5e5bbb7486f041b2db095ea1d4c0d829bc42b522e2009c85063d08"}
SURF_V2_TIMES = os.path.join(G.ROOT, "surface_v2", "surf_v2", "gapsw", "analysis", "gapsw_times.csv")
SURF_V2_CONFIG = os.path.join(G.ROOT, "surface_v2", "surf_v2", "config.json")
KIND_FE, KIND_TERM = "other", "other"    # the Grace-basis kinds of the front end and of the routed term (the stage's rule)
STATISTIC = ("the fastest of the term's three fresh processes (Decoder Switching is the baseline of this comparison; there is no "
             "'our side' in this family; STAGE2_common C6)")


def conv(us, kind, fit):
    """A Linux workstation time on the Grace basis (grace_fit.to_grace), None when the time or the stage's fit is missing."""
    return None if fit is None or us is None else grace_fit.to_grace(us, kind, fit)


def load_fit(path):
    """The stage's Grace fit (the surface family's fit_s2.json, whose "fit" holds grace_fit.fit's output, or that output alone) -> (fit, sha256).
    Refused unless it holds the keys grace_fit.to_grace reads and re-derives: grace_fit.fit on the raw time of every point gives
    the file's factors and `applied`. 'none' -> (None, None): the stage's fit was not established."""
    if path == "none":
        return None, None
    fit = json.load(open(path))
    if "fit" in fit and "points" not in fit:           # fit_s2.json (analysis_s.fit_s2): grace_fit.fit's output sits under "fit"
        if fit["fit"] is None:                         # complete false: the stage's fit was not established
            return None, None
        fit = fit["fit"]
    for k in ("points", "factors", "spread", "common", "applied"):
        assert k in fit, f"{path}: no key {k!r}: not a grace_fit.fit output"
    assert all(isinstance(fit["applied"].get(k), (int, float)) and fit["applied"][k] > 0 for k in grace_fit.KINDS), \
        f"{path}: applied has no positive factor for each of {grace_fit.KINDS}"
    pts = {(r["decoder"], r["d"], r["p"]): r["xeon_us_per_shot"] for r in fit["points"]}
    again = grace_fit.fit(pts)                         # raises when a point is missing
    for k in grace_fit.KINDS:
        assert math.isclose(again["applied"][k], fit["applied"][k], rel_tol=1e-9), f"{path}: applied[{k!r}] does not re-derive from its points"
    for k in ("plain", "corr", "common"):
        assert math.isclose(again["factors"][k], fit["factors"][k], rel_tol=1e-9), f"{path}: factors[{k!r}] does not re-derive"
    assert bool(again["common"]) == bool(fit["common"]), f"{path}: `common` does not re-derive"
    return fit, G.sha256_file(path)


def fmt(x, n=2):
    return "withheld" if x is None else f"{x:,.{n}f}"


def rederive(rec):
    """The host-check reasons the record's raw counters give (hostwin's limits), in the order the process writes them."""
    out = []
    if not H.shares(rec["pre_idle"]["start"], rec["pre_idle"]["end"], rec["core"])["ok"]:
        out.append("idle")
    p = rec["pre_probe"]
    if not H.probe_check(p["values_us"], p["ref_us"])["ok"]:
        out.append("probe")
    if "window" in rec and not H.shares(rec["window"]["start"], rec["window"]["end"], rec["core"])["ok"]:
        out.append("window")
    return out


def reasons_of(rec):
    return [r.split(":")[0] for r in rec["reasons"] if r.split(":")[0] in ("idle", "probe", "window")]


def check_record(rec, kind, sess, man, dry, masks, pol_sha):
    """An accepted process: everything the registration says it must satisfy (raises AssertionError otherwise)."""
    assert rec["schema"] == "gs-1" and rec["kind"] == kind and rec["accepted"] and not rec["reasons"], "not an accepted record"
    assert rec["pinned"] and rec["env"]["omp"] == "1", "pinning or threads"
    assert rec["gapsw_sha256"] == G.GAPSW_SHA256 and rec["env"]["cpu"] == sess["cpu"], "gapsw.py or the CPU"
    assert rec["limits"] == H.LIMITS, "the limits are not the registered ones"
    assert dry and rec.get("ignored_checks") or rederive(rec) == [], "the raw counters do not give an accepted process"
    assert rec["pre_probe"]["ref_us"] == sess["probe_ref_us"], "the probe's reference is not the session's"
    assert bool(rec["dry"]) == dry == bool(man.get("dry")), "dry flag"
    if not dry:
        assert rec["release_commit"] == sess["deployed"], "a process of another deployment"
    return rec


def load_session(path, man, dry):
    sess = json.load(open(os.path.join(path, "session.json")))
    att = json.load(open(os.path.join(path, "attempts.json")))
    assert "end" in att, "the session did not reach its end (attempts.json has no end): a technical failure"
    assert sess["manifest_sha256"] == G.sha256_file(man), "the session did not read this manifest"
    assert dry or sess["sets_sha256"] == G.sha256_file(G.SETS_FILE), "the session did not read this sets file"
    refs = sorted(k for k, e in att["slots"].items() if e["kind"] == "ref")      # only an older dry session has them
    assert dry or not refs, "a reference slot in a registered session (this family has none)"
    sess["ignored_reference_slots"] = refs
    sess["slots"] = [k for k in sess["slots"] if k not in refs]
    att["slots"] = {k: v for k, v in att["slots"].items() if k not in refs}
    plan = [s["id"] for s in drive_gs.plan(sess["reps"])]
    assert dry or (sess["slots"] == plan and sess["reps"] == G.REPS and not sess["dry"]), "not the registered plan"
    assert set(att["slots"]) == set(sess["slots"]), "attempts.json lists other slots than the session"
    return sess, att


def collect(path, sess, att, man, dry, masks_dir):
    """{slot: dict(status, rec (the accepted process's record or None), attempts: [...])}."""
    out = {}
    for sid in sess["slots"]:
        ent = att["slots"][sid]
        assert 1 <= len(ent["attempts"]) <= G.MAX_ATTEMPTS, f"{sid}: attempts"
        got = None
        for i, at in enumerate(ent["attempts"]):
            assert at["k"] == i + 1 and (got is None), f"{sid}: an attempt after the accepted one"
            f = os.path.join(path, at["dir"], "rec.json")
            rec = json.load(open(f)) if os.path.exists(f) else None
            assert rec is None or (rec["core"] == at["core"] and sess["core"] in (None, at["core"])), f"{sid} a{i + 1}: the core"
            if at["status"] == "accepted":
                got = check_record(rec, ent["kind"], sess, man, dry, masks_dir, None)
            elif at["status"] == "rejected" and at.get("window_passed") is False and at["rc"] == 0:
                # cpu_ws rejected the process's whole window by its reference check (STAGE2_common C5): the process passed its
                # own check and its record stays, but the attempt does not count
                assert rec is not None and rec["accepted"] and not rec.get("failed"), f"{sid} a{i + 1}: record"
            elif at["status"] == "rejected":
                assert rec is not None and not rec.get("failed") and not rec["accepted"], f"{sid} a{i + 1}: record"
                forced = dry and any("forced" in r for r in rec["reasons"])
                assert at["rc"] == 4 and reasons_of(rec) == rederive(rec) and (reasons_of(rec) or forced), \
                    f"{sid} a{i + 1}: the rejection is not re-derived from the counters"
            else:
                assert at["status"] == "failed", at
        # STAGE2_common C5: a slot without an accepted attempt is not established, whether its three attempts were rejected or
        # failed or a second loss of the Linux workstation cut it short; the session goes on
        status = "accepted" if got else "not established"
        out[sid] = dict(status=status, rec=got, attempts=[(a["k"], a["status"], a["reasons"]) for a in ent["attempts"]])
    return out


def median3(xs, reps=G.REPS):
    """The median of the repetitions (three in a registered session; a dry run may have fewer), None unless all are there."""
    return statistics.median(xs) if len(xs) == reps else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--prescreen", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--fit", required=True, help="the stage's Grace fit file (the surface family's analysis writes it); 'none': not established")
    ap.add_argument("--beacon", default="")
    ap.add_argument("--commits", nargs="*", default=[])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dry = a.dry
    man = G.manifest(a.manifest)
    assert bool(man.get("dry")) == dry, "--dry and the manifest disagree"
    from speed_rt import sets
    seeds = sets.all_seeds(man["beacon"]["randomness"], G.SETS_FILE)
    assert dry or set(man["sets"]) == set(G.sets()), "the manifest's sets are not the sets file's"
    for f, e in man["files"].items():
        if e["set"] not in seeds:                           # an older dry draw's reference sets (srt-gs-ref-*): not read
            assert dry and e["set"].startswith("srt-gs-ref-"), f"{f}: a set the sets file does not have"
            continue
        assert e["seed"] == seeds[e["set"]][e["j"]], f"{f}: seed is not the beacon's"
    beacon = None
    if not dry:
        from speed_rt import beacon as B
        beacon = B.load(a.beacon)
        assert beacon["randomness"] == man["beacon"]["randomness"] and beacon["round"] == man["beacon"]["round"], "manifest vs beacon"
        assert man["files"].keys() == {G.ds_file(d, j) for d in G.DS_DISTANCES for j in range(G.sets()[G.ds_set(d)]["chunks"])}, "the manifest is not the full draw"
    fit, fit_sha = load_fit(a.fit)
    sess, att = load_session(a.session, a.manifest, dry)
    assert dry or sess["deployed"] in a.commits, "the session ran from a deployment other than F or B"
    pre = json.load(open(os.path.join(a.prescreen, "prescreen.json")))
    assert pre["passed"] and sess["core"] in (None, pre["core"]) and pre["probe"]["ref_us"] == sess["probe_ref_us"], "pre-screen"
    assert pre["cpu"] == sess["cpu"], "the pre-screen and the session ran on different CPUs"
    slots = collect(a.session, sess, att, man, dry, a.masks)
    if dry:                                                 # a dry run may have run only some slots (drive_gs --only)
        for sl in drive_gs.plan(sess["reps"]):
            slots.setdefault(sl["id"], dict(status="not run", rec=None, attempts=[]))
    same = {}                                               # what every accepted process must share
    masks = {}
    for sid, r in sorted(slots.items()):
        x = r["rec"]
        if x:
            key = G.ds_set(x["d"])
            assert x["shots_sha256"] == man["shots_sha256"][key], f"{key}: the process did not read the draw's shots"
            for k in ("code_digest", "stage_sha256", "release_commit", "env"):
                v = json.dumps(x[k], sort_keys=True)
                assert same.setdefault(k, v) == v, f"{sid}: {k} differs from the other processes' of the session"
            if x["d"] not in masks:
                stem = os.path.join(a.masks, f"mask_d{x['d']}")
                mrec = json.load(open(stem + ".json"))
                assert mrec["npz_sha256"] == G.sha256_file(stem + ".npz"), f"mask d = {x['d']}: not the file its record names"
                assert mrec["shots_sha256"] == man["shots_sha256"][key] and mrec["gapsw_sha256"] == G.GAPSW_SHA256 and \
                    mrec["policy_sha256"] == REGISTERED_POLICY_SHA256[x["d"]], f"mask d = {x['d']}: not this draw's or this policy's"
                masks[x["d"]] = mrec
            m = masks[x["d"]]
            assert x["mask_record"] == m["npz_sha256"] and x["routed"] == m["routed"] and x["tau"] == m["tau"], \
                f"{sid}: the process did not read this mask"
    res = dict(what="SPEED-RT Decoder Switching stage: gap switching re-timed on the CPU host" + (" -- DRY RUN, never a result" if dry else ""),
               dry=dry, cpu=sess["cpu"], cgroup_cpu=sess["cgroup_cpu"], cpus_allowed=sess["cpus_allowed"], core=sess["core"], core_rule=sess.get("core_rule"),
               cores_used=sorted({v["rec"]["core"] for v in slots.values() if v["rec"]}),
               attempt_cores={k: [a["core"] for a in att["slots"][k]["attempts"]] for k in sorted(att["slots"])},
               node=sess["node"], deployed=sess["deployed"], beacon_round=man["beacon"]["round"], probe_ref_us=sess["probe_ref_us"],
               prescreen=dict(probe=pre["probe"], selftest=pre["selftest"], cgroup_cpu=pre["cgroup_cpu"]),
               session_seconds=att["seconds"], attempts={k: v["attempts"] for k, v in sorted(slots.items())},
               status={k: v["status"] for k, v in sorted(slots.items())})
    # ---- the stage's Grace fit (read, not measured here) and the conversion this family applies
    res["statistic"] = STATISTIC
    res["fit"] = fit
    res["fit_file"] = None if fit is None else dict(name=os.path.basename(a.fit), sha256=fit_sha)
    res["conversion"] = dict(front_end_kind=KIND_FE, routed_kind=KIND_TERM,
                             front_end_factor=None if fit is None else fit["applied"][KIND_FE],
                             routed_factor=None if fit is None else fit["applied"][KIND_TERM],
                             other_kind_factor=None if fit is None else fit["applied"]["other"],
                             common=None if fit is None else bool(fit["common"]))
    res["ignored_reference_slots"] = sess["ignored_reference_slots"]
    fit_rows = res["fit_rows"] = []
    if fit:
        fit_rows += [dict(decoder=r["decoder"], d=r["d"], p=r["p"], xeon_us_per_shot=r["xeon_us_per_shot"],
                         nvidia_us_per_round=grace_fit.NVIDIA_US_PER_ROUND[(r["decoder"], r["d"], r["p"])],
                         grace_us_per_shot=r.get("grace_us_per_shot"), ratio=r.get("ratio"), residual_ln=r.get("residual_ln")) for r in fit["points"]]
    # ---- Decoder Switching's terms
    old = {}
    if os.path.exists(SURF_V2_TIMES):
        import csv as _csv
        old = {int(r["d"]): r for r in _csv.DictReader(open(SURF_V2_TIMES))}
    cfg = json.load(open(SURF_V2_CONFIG))
    rows = []
    for d in G.DS_DISTANCES:
        fe = [slots[f"r{r}-fe-d{d}"]["rec"] for r in range(1, sess["reps"] + 1)]
        te = [slots[f"r{r}-term-d{d}"]["rec"] for r in range(1, sess["reps"] + 1)]
        fe_ok, te_ok = all(fe) and len(fe) == sess["reps"], all(te) and len(te) == sess["reps"]
        for x in (fe + te):
            if x:
                assert x["policy_sha256"] == REGISTERED_POLICY_SHA256[d], f"d = {d}: not the registered policy"
        fe_us = [x["term_us"] for x in fe if x]
        te_us = [x["term_us"] for x in te if x]
        pl_us = [x["plain_in_front_us"] for x in fe if x]
        row = dict(d=d, tau=fe[0]["tau"] if fe[0] else (te[0]["tau"] if te[0] else None),
                   fe_reps_us=fe_us, term_reps_us=te_us, plain_in_front_reps_us=pl_us,
                   fe_reps_grace_us=[conv(x, KIND_FE, fit) for x in fe_us], term_reps_grace_us=[conv(x, KIND_TERM, fit) for x in te_us],
                   fe_fastest_us=min(fe_us) if fe_ok else None, term_fastest_us=min(te_us) if te_ok else None,
                   fe_slowest_us=max(fe_us) if fe_ok else None, term_slowest_us=max(te_us) if te_ok else None,
                   fe_median_us=median3(fe_us, sess["reps"]) if fe_ok else None, term_median_us=median3(te_us, sess["reps"]) if te_ok else None,
                   plain_in_front_fastest_us=min(pl_us) if fe_ok else None, plain_in_front_slowest_us=max(pl_us) if fe_ok else None,
                   routed=[x["routed"] for x in fe + te if x], routed_fraction=(fe[0] or te[0] or {}).get("routed_fraction"),
                   conditional_mean_us=[x["conditional_mean_us"] for x in te if x],
                   fe_se_rel=[x["term_sd_us"] / math.sqrt(x["shots"]) / x["term_us"] for x in fe if x and "term_sd_us" in x],
                   term_se_rel=[x["term_sd_us"] / math.sqrt(x["shots"]) / x["term_us"] for x in te if x and "term_sd_us" in x and x["term_us"]])

        def comp(f, t, kf, kt):                  # a composite: front end f + routed term t (Linux workstation us), each on its kind's basis
            if f is None or t is None:
                return None
            return f + t if kf is None else (None if fit is None else conv(f, kf, fit) + conv(t, kt, fit))
        for stat in ("fastest", "slowest", "median"):    # the raw composites (the Ryzen basis, UPDATE 27 item 8)
            row[f"raw_composite_{stat}_us"] = comp(row[f"fe_{stat}_us"], row[f"term_{stat}_us"], None, None)
        row["raw_composite_us"] = row["raw_composite_fastest_us"]          # the row's value: the fastest of three
        row["fe_grace_us"] = conv(row["fe_fastest_us"], KIND_FE, fit)
        row["term_grace_us"] = conv(row["term_fastest_us"], KIND_TERM, fit)
        row["plain_in_front_grace_us"] = conv(row["plain_in_front_fastest_us"], "plain", fit)
        row["composite_grace_us"] = comp(row["fe_fastest_us"], row["term_fastest_us"], KIND_FE, KIND_TERM)
        row["composite_slowest_grace_us"] = comp(row["fe_slowest_us"], row["term_slowest_us"], KIND_FE, KIND_TERM)        # sensitivity
        row["composite_median_grace_us"] = comp(row["fe_median_us"], row["term_median_us"], KIND_FE, KIND_TERM)          # sensitivity
        row["composite_pm_kinds_grace_us"] = comp(row["fe_fastest_us"], row["term_fastest_us"], "plain", "corr")         # sensitivity
        row["composite_fe_corr_grace_us"] = comp(row["fe_fastest_us"], row["term_fastest_us"], "corr", "corr")           # sensitivity
        row["front_over_plain"] = (row["fe_fastest_us"] / row["plain_in_front_fastest_us"]) if fe_ok else None
        o = old.get(d)
        row["surf_v2"] = None if o is None else dict(front_us=float(o["gapsw_front_us"]), routed_us=float(o["gapsw_routed_us"]),
                                                     composite_us=float(o["gapsw_composite_us"]),
                                                     projected_us=float(o["gapsw_projected_us"]))
        reg = cfg["registered_times_us"][str(d)]
        row["paper_basis"] = dict(plain_pm_us=reg["plain_pm"], corr_pm_us=reg["corr_pm"])
        rows.append(row)
    res["rows"] = rows
    # context: the front end against plain PM at d = 13 on the stage's reference point (the fit's raw time of plain PM, d = 13, p = 0.6 %:
    # the slowest of three of the surface family's reference session, not a process of this family's session)
    pt = {(r["decoder"], r["d"], r["p"]): r for r in fit["points"]} if fit else {}
    p13 = pt.get(("plain", 13, 0.006), {}).get("xeon_us_per_shot")
    r13 = next(r for r in rows if r["d"] == 13)
    res["front_over_plain_pm_d13"] = dict(plain_pm_fit_point_us=p13, plain_pm_fit_point_statistic="slowest of three (the stage's fit)",
                                          front_end_fastest_us=r13["fe_fastest_us"], front_end_slowest_us=r13["fe_slowest_us"],
                                          ratio_fastest=(r13["fe_fastest_us"] / p13) if p13 and r13["fe_fastest_us"] else None,
                                          ratio_slowest=(r13["fe_slowest_us"] / p13) if p13 and r13["fe_slowest_us"] else None)
    res["words"] = words(res, fit)
    os.makedirs(a.out, exist_ok=False)
    with open(os.path.join(a.out, "gs_numbers.json"), "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    write_csv(os.path.join(a.out, "gs_fit.csv"), fit_rows, ("decoder", "d", "p", "xeon_us_per_shot", "nvidia_us_per_round",
                                                          "grace_us_per_shot", "ratio", "residual_ln"))
    write_csv(os.path.join(a.out, "gs_times.csv"), rows, ("d", "tau", "fe_reps_us", "fe_reps_grace_us", "term_reps_us", "term_reps_grace_us",
                                                         "fe_fastest_us", "term_fastest_us", "raw_composite_us", "fe_grace_us", "term_grace_us",
                                                         "composite_grace_us", "fe_slowest_us", "term_slowest_us", "raw_composite_slowest_us",
                                                         "composite_slowest_grace_us", "fe_median_us", "term_median_us", "raw_composite_median_us",
                                                         "composite_median_grace_us", "composite_pm_kinds_grace_us", "composite_fe_corr_grace_us",
                                                         "routed_fraction", "front_over_plain"))
    with open(os.path.join(a.out, "gs_tables.md"), "x") as fh:
        fh.write(tables(res, fit))
    print(f"outputs in {a.out}: " + ", ".join(sorted(os.listdir(a.out))))
    for w in res["words"]:
        print(w)


def write_csv(path, rows, cols):
    with open(path, "x", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([json.dumps(r.get(c)) if isinstance(r.get(c), (list, dict)) else ("" if r.get(c) is None else repr(r.get(c)))
                        for c in cols])


def words(res, fit):
    """The descriptive rows' sentences (the registered wording)."""
    cpu = res["cpu"]
    out = []
    if fit:
        out.append(f"GS-fit: On the host's {cpu}, NVIDIA's Grace time over ours for plain PM is {fit['factors']['plain']:.4f} "
                   f"(spread {fit['spread']['plain']:.4f}), for correlated PM {fit['factors']['corr']:.4f} "
                   f"(spread {fit['spread']['corr']:.4f}); {'one common factor ' + format(fit['factors']['common'], '.4f') + ' applies' if fit['common'] else 'each decoder keeps its own factor'}"
                   f" (the stage's fit, read from {res['fit_file']['name']}). Gap switching's front end and routed term are converted as kind "
                   f"'other' (factors {fit['applied'][KIND_FE]:.4f} and {fit['applied'][KIND_TERM]:.4f}).")
    else:
        out.append("GS-fit: the stage's Grace fit was not established in SPEED-RT; no value is converted to the Grace basis, and no "
                   "earlier value stands in for it.")
    for r in res["rows"]:
        d = r["d"]
        if r["composite_grace_us"] is not None:
            out.append(f"GS-{d}: On the {cpu}, gap switching's front end took {r['fe_fastest_us']:.2f} us per shot at d = {d} "
                       f"({r['fe_grace_us']:.2f} us on the Grace basis), correlated PM on its routed shots {r['term_fastest_us']:.2f} us "
                       f"({r['term_grace_us']:.2f} us), and the composite {r['composite_grace_us']:.2f} us (each term the fastest of three "
                       f"processes; SURF-v2's earlier composite on the EPYC 7713 basis: {fmt(r['surf_v2'] and r['surf_v2']['composite_us'])} us).")
        else:
            out.append(f"GS-{d}: Gap switching's composite at d = {d} was not established on the {cpu} in SPEED-RT "
                       f"({'front end' if r['fe_fastest_us'] is None else 'routed term' if r['term_fastest_us'] is None else 'Grace fit'} "
                       "missing); the comparison that needs it is withheld, and no earlier value stands in for it.")
    return out


def tables(res, fit):
    L = [f"# SPEED-RT Decoder Switching stage{' (DRY RUN: never a result)' if res['dry'] else ''}", "",
         f"Host CPU: {res['cpu']} (container quota {res['cgroup_cpu']}, {res['cpus_allowed']} CPUs allowed), timed cores "
         f"{'all ' + str(res['core']) if res['core'] is not None else 'picked per attempt, used: ' + ', '.join(map(str, res['cores_used']))}, "
         f"beacon round {res['beacon_round']}, deployed {res['deployed']}, session {res['session_seconds']:.0f} s. "
         f"Host probe reference (plain PM per block, d = 9): {res['probe_ref_us']:.3f} us per shot.", "",
         f"Statistic: {res['statistic']}. The slowest and the median of three are sensitivity rows. Every raw Linux workstation time (Ryzen basis) "
         "stands beside its Grace-basis value.", "",
         "## The Grace fit (the stage's, as read from the fit file; this family times no reference points)", ""]
    if fit:
        L += [f"Fit file {res['fit_file']['name']} (sha256 {res['fit_file']['sha256']}).", "",
              "| decoder | d | p | Ryzen time (the fit's: slowest of three) | NVIDIA (Grace) | ratio | residual (ln) |", "|---|---|---|---|---|---|---|"]
        for r in res["fit_rows"]:
            L.append(f"| {r['decoder']} | {r['d']} | {r['p']} | {fmt(r['xeon_us_per_shot'], 3)} | {r['nvidia_us_per_round'] * r['d']:.2f} | "
                     f"{fmt(r['ratio'], 4)} | {fmt(r['residual_ln'], 4)} |")
        f, s = fit["factors"], fit["spread"]
        L += ["", f"Factors (geometric means): plain {f['plain']:.4f} (spread {s['plain']:.4f}), correlated {f['corr']:.4f} "
                  f"(spread {s['corr']:.4f}), common {f['common']:.4f}; applied: {'the common factor' if fit['common'] else 'each its own'} "
                  f"(plain {fit['applied']['plain']:.4f}, correlated {fit['applied']['corr']:.4f}, other {fit['applied']['other']:.4f}). "
                  f"Rule: {fit['rule']}. This family converts the front end with the plain-PM factor and the routed term with the "
                  f"correlated-PM factor (grace_fit.to_grace)."]
    else:
        L += ["The stage's fit was not established: no value below is converted (withheld)."]
    L += ["", "## Gap switching, per distance (us per shot; each term the fastest of three repetitions; Ryzen = raw Linux workstation time)", "",
          "| d | routed (timing shots) | front end, Ryzen | its plain-PM decode, Ryzen | front end / plain PM | routed PM, Ryzen | raw composite | "
          "front end, Grace | routed PM, Grace | composite, Grace | SURF-v2's earlier composite (EPYC 7713 basis) |", "|" + "---|" * 11]
    for r in res["rows"]:
        L.append(f"| {r['d']} | {fmt(100 * r['routed_fraction'] if r['routed_fraction'] is not None else None, 2)} % | "
                 f"{fmt(r['fe_fastest_us'])} | {fmt(r['plain_in_front_fastest_us'])} | {fmt(r['front_over_plain'])} | "
                 f"{fmt(r['term_fastest_us'])} | {fmt(r['raw_composite_us'])} | {fmt(r['fe_grace_us'])} | {fmt(r['term_grace_us'])} | "
                 f"{fmt(r['composite_grace_us'])} | {fmt(r['surf_v2'] and r['surf_v2']['composite_us'])} |")
    L += ["", "## Sensitivity rows (deciding nothing; composites, us per shot)", "",
          "| d | fastest, raw | fastest, Grace | slowest, raw | slowest, Grace | median, raw | median, Grace | fastest, front end plain and routed corr | "
          "fastest, front end with the correlated factor |", "|" + "---|" * 9]
    for r in res["rows"]:
        L.append(f"| {r['d']} | {fmt(r['raw_composite_fastest_us'])} | {fmt(r['composite_grace_us'])} | {fmt(r['raw_composite_slowest_us'])} | "
                 f"{fmt(r['composite_slowest_grace_us'])} | {fmt(r['raw_composite_median_us'])} | {fmt(r['composite_median_grace_us'])} | "
                 f"{fmt(r['composite_pm_kinds_grace_us'])} | {fmt(r['composite_fe_corr_grace_us'])} |")
    L += ["", "## Every repetition (us per shot; raw, then Grace basis)", "", "| d | front end, raw | front end, Grace | routed PM, raw | routed PM, Grace |",
          "|---|---|---|---|---|"]

    def lst(xs, n=2):
        return ", ".join(fmt(x, n) for x in xs) or "none"
    for r in res["rows"]:
        L.append(f"| {r['d']} | {lst(r['fe_reps_us'])} | {lst(r['fe_reps_grace_us'])} | {lst(r['term_reps_us'])} | {lst(r['term_reps_grace_us'])} |")
    se = [(r["d"], max(r["fe_se_rel"], default=None), max(r["term_se_rel"], default=None)) for r in res["rows"]]
    L += ["", "Sampling error of the per-shot means (relative standard error, the larger of three repetitions; front end / routed term): " +
          "; ".join(f"d = {d}: {'n/a' if f is None else format(100 * f, '.3f') + ' %'} / {'n/a' if t is None else format(100 * t, '.3f') + ' %'}"
                    for d, f, t in se) + "."]
    x = res["front_over_plain_pm_d13"]
    L += ["", f"Front end over plain PM at d = 13, on the stage's reference point (the fit's raw time of plain PM, p = 0.6 %, the slowest of three "
              f"of the reference session; not a process of this family's session): plain PM {fmt(x['plain_pm_fit_point_us'])} us, front end "
              f"{fmt(x['front_end_fastest_us'])} us (fastest of three; ratio {fmt(x['ratio_fastest'])}) and {fmt(x['front_end_slowest_us'])} us "
              f"(slowest of three; ratio {fmt(x['ratio_slowest'])}).", "", "## The rows", ""]
    L += [f"- {w}" for w in res["words"]]
    bad = sorted(k for k, v in res["status"].items() if v != "accepted")
    L += ["", f"Slots not established: {bad if bad else 'none'}. Attempts beyond the first: " +
          (", ".join(f"{k} ({len(v)})" for k, v in sorted(res['attempts'].items()) if len(v) > 1) or "none") + "."]
    if res["ignored_reference_slots"]:
        L += ["", f"{len(res['ignored_reference_slots'])} reference slots of an older dry session were not read (this family has none)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
