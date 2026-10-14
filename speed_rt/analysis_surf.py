"""The surface stage's analysis (the stage's section, "Analysis"): from the two pods' records to every surface speed number the
paper prints, old -> new, and every surface speed comparison's verdict, by the frozen rules; GPU terms per pod (S-low: d = 7, 9, 11;
S-high: d = 13, 15, 17), every CPU time from the Linux workstation (the CPU worker), converted to the Grace basis by the stage's ONE fit
(speed_rt/grace_fit.py on the 18 reference points, the slowest of three each; OUT/fit_s2.json). Deterministic (no clock, no
randomness, sorted keys): it runs twice and the two outputs must be byte-identical.
    python -m speed_rt.analysis_surf --records RECORDS --out OUT [--dry] [--fit FIT_S2.json]
    python -m speed_rt.analysis_surf fit --cpu RECORDS/cpu --testsets RECORDS/testsets --out fit_s2.json [--dry]
        the Grace fit alone, from the CPU worker's C-ref records (cpu_attempts.json, cpu_session.json, cpu/) and srt-ref.json; no pod record is read, the
        CPU session need not be finished; exit 0 when the fit is complete (all 18 points), 3 otherwise (the file then names the missing points).
        --fit on the full analysis: the file must equal the fit this run derives (the analysis stops otherwise).
RECORDS holds, as the pods wrote them (speed_rt/pod/collect_s.sh copies them there):
  low/{session,masks,res}      pod S-low: SESSION (speed_rt/pod_s.py: attempts.json, META.json, DONE.json, prescreen.json, every process's
  high/{session,masks,res}     record), its routing masks and its Ising residuals; pod S-high alike
  cpu/{cpu_session.json,cpu_attempts.json,cpu/}   the CPU worker's session on the Linux workstation (section 1.14: the records of speed_rt.gref time
                               and timing.cpu_term ref|term|share, one directory per attempt, cpu_attempts.json as pod_s's attempts.json)
  testsets/srt-surf.json       the draw's manifests (speed_rt/draw.py, committed as part of commit B)
  testsets/srt-ref.json
OUT/analysis.json: every quantity with every repetition and every reason a quantity is missing, the stage's fit (every point,
the factors, the spreads, the residuals; also OUT/fit_s2.json), the bases, the table and the sha256 of every input file; OUT/table.md: the results
table; OUT/rows.csv: one line per row. Checks the records against the manifests, speed_rt/circuits/ref_circuits.json and
freeze_surf.json; a pod session that did not reach its end (or whose recheck failed) establishes nothing, and every comparison of
that pod is withheld."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "surface_v2"))

from speed_rt import analysis_s as AS, table_s as TS  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def inputs_digest(records):
    out = {}
    for d, _, fs in sorted(os.walk(records)):
        for f in sorted(fs):
            p = os.path.join(d, f)
            out[os.path.relpath(p, records)] = sha256_file(p)
    return out


def merge(roles, key):
    out = {}
    for r in roles.values():
        out.update(r[key])
    return out


def analyse(records, dry, fit_file=None):
    cfg = json.load(open(os.path.join(HERE, "config_surf.json")))
    freeze = json.load(open(os.path.join(HERE, "freeze_surf.json")))
    circuits = json.load(open(os.path.join(HERE, "circuits", "ref_circuits.json")))["circuits"]
    man_s = json.load(open(os.path.join(records, "testsets", "srt-surf.json")))
    man_r = json.load(open(os.path.join(records, "testsets", "srt-ref.json")))
    assert bool(man_s.get("dry")) == dry == bool(man_r.get("dry")), "--dry and the draw's manifests disagree"
    roles = {r: AS.role(r, os.path.join(records, r), os.path.join(records, "cpu"), freeze, cfg, man_s, man_r, circuits, dry)
             for r in AS.ROLES}
    cpu = os.path.join(records, "cpu")
    ok_c, _ = AS.qualifies_cpu(cpu, dry)
    f2 = AS.fit_s2(cpu, freeze, man_r, circuits, dry) if ok_c else None
    if fit_file:
        ff = json.load(open(fit_file))
        if f2 is None:                                     # the CPU session did not reach its end: the fit file must say so (fit_main)
            assert ff.get("fit") is None and ff.get("complete") is False, f"{fit_file} holds a fit, but the CPU session does not qualify"
        else:
            assert json.dumps(ff, sort_keys=True) == json.dumps(f2, sort_keys=True), f"{fit_file} is not the fit derived from these records"
    for r in roles.values():
        assert not r["qualifies_cpu"] or r["fit"] == (f2 or {}).get("fit"), "a role's fit differs from the stage's fit"
    dep = {r["meta"]["deployed"] for r in roles.values() if r["qualifies"]}
    assert dry or len(dep) <= 1, f"the two pods ran different trees: {sorted(dep)}"
    Sq, N, why, reps = (merge(roles, k) for k in ("Sq", "N", "why", "reps"))
    comp = {b: {} for b in ("grace", "xeon")}
    base = {b: {} for b in ("grace", "xeon")}
    for r in roles.values():
        for b in comp:
            comp[b].update(r["comp"][b])
            base[b].update(r["base"][b])
    parts = merge(roles, "parts")
    nets = {s: {} for s, _ in TS.SETS}
    for name, n in freeze["surface"]["networks"].items():
        nets["sv2" if n["role"] == "new" else "deployed"][n["d"]] = name
    fits = {k: r["fit"] for k, r in roles.items()}
    T = TS.table(comp, parts, base, Sq, N, nets, why, fits)
    q = dict(pods={k: dict(qualifies=r["qualifies"], qualifies_cpu=r["qualifies_cpu"], meta=r["meta"], cpu_meta=r["cpu_meta"], fit=r["fit"], factors_for_b=r["B"]["grace"]["term_factor"],
                           bases={b: {k2: v for k2, v in sorted(x.items())} for b, x in r["B"].items()})
                   for k, r in roles.items()},
             session=Sq, xeon=N, composites=comp, parts=parts,
             baselines={b: {f"{k}.{d}": v for (k, d), v in sorted(x.items())} for b, x in base.items()},
             repetitions=reps, missing=why)
    return T, q, cfg, freeze, f2


def fit_main(argv):
    ap = argparse.ArgumentParser(prog="analysis_surf fit")
    ap.add_argument("--cpu", required=True, help="the CPU worker's records: cpu_session.json, cpu_attempts.json, cpu/")
    ap.add_argument("--testsets", required=True, help="the directory holding srt-ref.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    freeze = json.load(open(os.path.join(HERE, "freeze_surf.json")))
    circuits = json.load(open(os.path.join(HERE, "circuits", "ref_circuits.json")))["circuits"]
    man_r = json.load(open(os.path.join(a.testsets, "srt-ref.json")))
    assert bool(man_r.get("dry")) == a.dry, "--dry and the draw's manifest disagree"
    ok_c, why_c = AS.qualifies_cpu(a.cpu, a.dry)          # as analyse(): no fit from a CPU session that did not reach its end
    f2 = AS.fit_s2(a.cpu, freeze, man_r, circuits, a.dry) if ok_c else dict(
        what="SPEED-RT stage 2's Grace fit: not established, the CPU session does not qualify" + (" - DRY: never a result" if a.dry else ""),
        dry=a.dry, complete=False, missing={"cpu_session": why_c}, fit=None)
    with open(a.out, "x") as fh:
        json.dump(f2, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    print(f"{a.out}: {'complete' if f2['complete'] else 'INCOMPLETE ' + str(sorted(f2['missing']))}; "
          + ("" if not f2["complete"] else f"factors {f2['fit']['factors']}, common {f2['fit']['common']}"))
    raise SystemExit(0 if f2["complete"] else 3)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        fit_main(sys.argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true", help="dry records (a dry draw, --limit runs): never the results")
    ap.add_argument("--fit", default="", help="fit_s2.json (the fit command's output): must equal the fit these records give")
    a = ap.parse_args()
    T, q, cfg, freeze, f2 = analyse(a.records, a.dry, a.fit or None)
    out = dict(what="SPEED-RT surface stage's analysis (speed_rt/analysis_surf.py)" + (" - DRY: never a result" if a.dry else ""),
               dry=a.dry, config_sha256=sha256_file(os.path.join(HERE, "config_surf.json")),
               freeze_sha256=sha256_file(os.path.join(HERE, "freeze_surf.json")),
               ref_circuits_sha256=sha256_file(os.path.join(HERE, "circuits", "ref_circuits.json")),
               inputs=inputs_digest(a.records), quantities=q, table=T)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "analysis.json"), "x") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")
    if f2 is not None:
        with open(os.path.join(a.out, "fit_s2.json"), "x") as fh:
            json.dump(f2, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
    md, rows = TS.render(T)
    with open(os.path.join(a.out, "table.md"), "x") as fh:
        fh.write(md)
    with open(os.path.join(a.out, "rows.csv"), "x") as fh:
        fh.write(rows)
    cmp_ = [r for r in T if r["kind"] == "comparison" and not r.get("descriptive")]
    print(f"{len(T)} rows; {len(cmp_)} comparisons: {sum(r['holds'] is True for r in cmp_)} hold, "
          f"{sum(r['holds'] is False for r in cmp_)} do not, {sum(r['holds'] is None for r in cmp_)} withheld")


if __name__ == "__main__":
    main()
