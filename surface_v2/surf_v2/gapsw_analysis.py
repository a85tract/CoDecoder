"""The secondary comparison of the SURF-v2 evaluation (surf_v2/GAPSW_REGISTRATION.md): gap switching (surf_v2/gapsw.py;
Decoder Switching with plain PM in front and correlated PM behind) at its frozen tau against the ConvLines pair of each
distance, on the test set. Run once after surf_v2.analysis, from the same frozen tree, and a second time only to show
byte-identical outputs.

Descriptive: the failures of gap switching, the ConvLines pair, correlated PM and plain PM with Wilson 95 % intervals;
the failures of gap switching at every q of the grid (tau_q from the calibration shots, the frozen policy's table), the
ConvLines pair's own q marked; the composite times on the Grace basis (gap switching: the slowest of three repetitions
of the front end's term plus the slowest of three of its CPU term, each times G(d) / the median of the same job's three
refs; the ConvLines pair's composite from surf_v2.analysis's analysis.json, shown beside).
One test per distance, a family of its own (family G): the exact two-sided McNemar test of the gap-switching pair against
the ConvLines pair (surf_v2.analysis's row network at its frozen tau), the Holm step-down over the distances at 0.05; a
rejected test is "ConvLines pair fewer failures" or "gap switching fewer failures" by the discordant counts, any other
"inconclusive" (never "no difference"). The Newcombe 95 % interval of the difference is shown beside, pointwise.

It refuses: a config that is not the frozen one, a beacon record or test seed that does not check (surf_v2.analysis
check_beacon), a front-end read, policy or timing record made by another gapsw.py than this tree's, a front-end read
whose answers are not plain PM's test read's, a policy that is not the tree's (or, dry, --policies') or whose choices
are not those of config.json's q rule on its own table, a timing directory that is not one Slurm job's run from a
deployment of the freeze commit F or of B (its job.json, written once at the job's end, names the job and the deployed
commit and lists every file of the directory with its sha256; each ref and CPU term names that job, each front-end term
that commit, and outside dry runs every record the frozen read path), and every read surf_v2.analysis refuses (the same
checks, imported).

    python -m surf_v2.gapsw_analysis --freeze surf_v2/freeze.json --reads READS... --shots TEST --cpu GSCPU
        --main ANALYSIS/analysis.json --out OUT
    python -c "from surf_v2 import gapsw_analysis; gapsw_analysis.write_job('GSCPU')"   (the timing job's last step)
--dry: dry runs only (the dry set and freeze, stand-in reads allowed; never a result)."""
import argparse
import glob
import json
import math
import os

import numpy as np

from convlines import provenance, stats, testset
from surf_v2 import analysis, gapsw
from timing import cpu_term

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "config.json")
ALPHA = 0.05
Z95 = 1.959963984540054


def scan_gs(dirs):
    """{(set, d, "gapsw"): [(record, npz path)]} of the front end's reads under dirs."""
    out = {}
    for d in dirs:
        for j in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            r = json.load(open(j))
            if r.get("what", "").startswith("SURF-v2 gap-switching read"):
                out.setdefault((r["set"], r["d"], r["decoder"]), []).append((r, j[:-5] + ".npz"))
    return out


def rule_choice(pol, M, bar_us, fallback):
    """The q of config.json's rule on a policy's own table (surf_v2.qrule.fit_rule): the smallest admissible q within
    the limit, else the fallback ("argmin" or "largest"); None when no q is admissible. -> (with the bar, without it)."""
    lim = M - 2 * math.sqrt(2 * M)

    def choose(adm):
        ok = [x for x in adm if x["failures"] <= lim]
        if not adm:
            return None
        return (ok[0] if ok else min(adm, key=lambda x: (x["failures"], x["q"])) if fallback == "argmin" else adm[-1])["q"]
    return choose([x for x in pol["table"] if x["projected_us"] <= bar_us]), choose(pol["table"])


def rate(k, n):
    lo, hi = stats.wilson(k, n, Z95)
    return dict(failures=int(k), shots=int(n), ler=k / n, lo95=lo, hi95=hi)


def job_files(cdir):
    return sorted(os.path.relpath(p, cdir) for p in glob.glob(os.path.join(cdir, "**", "*"), recursive=True)
                  if os.path.isfile(p) and os.path.relpath(p, cdir) != "job.json")


def write_job(cdir):
    """The timing job's record, written once as its last step (alpine_gapsw_cpu.txt): the Slurm job, the deployed commit,
    the code, and every file the job wrote in cdir with its sha256."""
    rec = dict(what="SURF-v2 gap-switching timing job (surf_v2/args/alpine_gapsw_cpu.txt)",
               slurm_job=os.environ["SLURM_JOB_ID"], deployed=provenance.commit(), code_digest=provenance.code_digest(),
               gapsw_sha256=gapsw.gapsw_sha256(), files={f: analysis.sha_file(os.path.join(cdir, f)) for f in job_files(cdir)})
    with open(os.path.join(cdir, "job.json"), "x") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
    print(f"{cdir}/job.json: job {rec['slurm_job']}, deployed {rec['deployed']}, {len(rec['files'])} files", flush=True)


def job(cdir, allowed, fz, own, dry, din):
    """The timing directory is one completed Slurm job's, run from a deployment in `allowed` (F or B; dry: this run's):
    job.json lists exactly its files with their sha256, every ref and CPU term names that job, every front-end term that
    commit and this gapsw.py, and outside dry runs every record carries the frozen read path's digest."""
    f = os.path.join(cdir, "job.json")
    assert os.path.exists(f), f"{cdir}: no job.json, so no timing job completed there"
    j = json.load(open(f))
    din["gapsw_cpu/job.json"] = analysis.sha_file(f)
    assert j.get("slurm_job") and j.get("deployed") and j["deployed"] in allowed, \
        f"timing job {j.get('slurm_job')} ran from {j.get('deployed')}, not from a deployment of F or B"
    assert job_files(cdir) == sorted(j["files"]), f"{cdir}: not exactly the files its job wrote"
    for x in j["files"]:
        assert analysis.sha_file(os.path.join(cdir, x)) == j["files"][x], f"{x}: not the file its job wrote"
        if x.endswith(".json"):
            r = json.load(open(os.path.join(cdir, x)))
            if x.startswith(f"fe_{gapsw.KEY}_"):
                assert r["release_commit"] == j["deployed"] and r["gapsw_sha256"] == own, f"{x}: another deployment's"
            else:
                assert str(r.get("slurm_job")) == str(j["slurm_job"]), f"{x}: written by job {r.get('slurm_job')}"
            assert dry or r.get("code_digest") == fz["read_code_digest"], f"{x}: another read path's code"
    return dict(slurm_job=j["slurm_job"], deployed=j["deployed"])


def times(cdir, d, pol, own, dry, din):
    """Gap switching's composite on the Grace basis from the CPU directory (see the module's docstring)."""
    def reps(pattern):
        fs = sorted(glob.glob(os.path.join(cdir, pattern)))
        assert len(fs) == 3 or (dry and fs), f"{cdir}: {pattern} has {len(fs)} repetitions, not 3"
        for f in fs:
            din["gapsw_cpu/" + os.path.basename(f)] = analysis.sha_file(f)
        return [json.load(open(f)) for f in fs]
    ref, fe, term = reps(f"ref_d{d}_rep*.json"), reps(f"fe_{gapsw.KEY}_d{d}_rep*.json"), reps(f"term_{gapsw.KEY}_d{d}_rep*.json")
    assert all(r["kind"] == "correlated" for r in ref), "the refs are correlated PM's"
    assert all(r["gapsw_sha256"] == own for r in fe), "a front-end term of another gapsw.py"
    assert all(r["network"] == f"{gapsw.KEY}_d{d}" and r["tau"] == pol["tau"] for r in term), \
        "a CPU term of another mask than the frozen tau's"
    assert len({r["shots_sha256"] for r in fe + term}) == 1, "the terms were timed on different shots"
    val = json.load(open(os.path.join(ROOT, "routing", "manifest.json")))["val"][str(d)]["shots_sha256"]
    assert fe[0]["shots_sha256"] == val or dry, "the terms did not read the studies' validation shots"
    g, src = cpu_term.grace_numerator(d)
    node = float(np.median([r["per_block_us"] for r in ref]))
    ratio = g / node
    fe_us, term_us = max(r["term_us"] for r in fe), max(r["term_us"] for r in term)
    return dict(ref_reps_us=[r["per_block_us"] for r in ref], node_median_us=node, grace_numerator_us=g,
                grace_numerator_source=src, ratio=ratio, fe_reps_us=[r["term_us"] for r in fe],
                term_reps_us=[r["term_us"] for r in term], front_grace_us=fe_us * ratio, routed_grace_us=term_us * ratio,
                composite_us=(fe_us + term_us) * ratio, routed_validation=term[0]["routed"],
                routed_fraction_validation=term[0]["routed_fraction"], reps=len(fe))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", default=os.path.join(HERE, "freeze.json"))
    ap.add_argument("--reads", nargs="+", required=True, help="the networks', PM's and the front end's read directories")
    ap.add_argument("--shots", required=True, help="the test set's chunk files (dry: the dry set's)")
    ap.add_argument("--cpu", required=True, help="the front end's CPU directory (refs, fe and term repetitions)")
    ap.add_argument("--main", default="", help="surf_v2.analysis's analysis.json (the ConvLines pairs' composites)")
    ap.add_argument("--policies", default=gapsw.POLICIES, help="dry runs only: another policy directory")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(CONFIG))
    fz = json.load(open(a.freeze))
    assert fz["config_sha256"] == analysis.sha_file(CONFIG), "surf_v2/config.json is not the frozen one"
    assert bool(fz.get("dry")) == a.dry, "a dry freeze needs --dry, and only a dry freeze takes it"
    assert a.dry or os.path.abspath(a.policies) == os.path.abspath(gapsw.POLICIES), "--policies is for dry runs"
    own = gapsw.gapsw_sha256()
    tset = cfg["dry"]["set"] if a.dry else cfg["test"]["set"]
    exc = {int(k): v for k, v in fz.get("exceptions", {}).items()}
    reads, gs = analysis.scan(a.reads), scan_gs(a.reads)
    din = {"freeze.json": analysis.sha_file(a.freeze), "config.json": analysis.sha_file(CONFIG),
           "surf_v2/gapsw.py": own, f"testsets/{tset}.json": analysis.sha_file(os.path.join(ROOT, testset.SETS[tset]))}
    bk = None if a.dry else analysis.check_beacon(tset, din)
    main_res = None
    if a.main:
        main_res = json.load(open(a.main))
        assert main_res["inputs"]["freeze.json"] == din["freeze.json"], "--main was made from another freeze record"
        din["main analysis.json"] = analysis.sha_file(a.main)
    allowed = {bk["freeze_commit"], provenance.commit()} if bk else {provenance.commit()}   # F, and B (step 14g's)
    res = dict(what="SURF-v2's secondary comparison: gap switching against the ConvLines pairs (surf_v2/gapsw_analysis.py)"
               + (" -- DRY RUN, never a result" if a.dry else ""), dry=a.dry, freeze_commit=bk["freeze_commit"] if bk else None,
               beacon=bk, test_set=tset, distances={}, inputs=din, alpha_familywise=ALPHA,
               timing_job=job(a.cpu, allowed, fz, own, a.dry, din))
    tests, curves, trows = [], [], []
    for d in cfg["distances"]:
        k = str(d)
        pol, psha = gapsw.policy(d, a.policies)
        din[f"gapsw policy_d{d}.json"] = psha
        assert pol["d"] == d and pol["gapsw_sha256"] == own and bool(pol["dry"]) == a.dry, f"policy of d = {d}: not this tree's"
        q = cfg["qrule"]                     # the rule's settings and choice, not config.json's sha256 or text
        fb = q["fallback"][k] if isinstance(q["fallback"], dict) else q["fallback"]
        got = tuple((pol[w]["chosen"] or {}).get("q") for w in ("registered", "no_bar"))
        assert (pol["grid"], pol["margin"], pol["bar_us"]) == (sorted(q["grid"][k]), q["M"][k], q["bar_us"][k]) and \
            rule_choice(pol, q["M"][k], q["bar_us"][k], fb) == got, f"policy of d = {d}: not the choice of config.json's q rule"
        assert pol["tau"] is not None, f"policy of d = {d}: {pol['outcome']}"
        ch = testset.parse_range(cfg["dry"]["test_chunks"] if a.dry else cfg["test"]["chunks"][k])
        obs = analysis.set_obs(tset, d, ch, a.shots)
        dec = {}
        for key in ("plain_pm", "corr_pm"):
            x, recs = analysis.merged(reads, (tset, d, key), ch, din)
            assert np.array_equal(x["obs"], obs), f"d{d} {key}: observables differ from the test set's"
            analysis.check_opp(recs, fz, a.dry)
            dec[key] = x["dec"]
        nets = cfg["networks"][k]
        name = nets["deployed"] if d in exc else nets["new"]
        fn = fz["networks"][name]
        x, recs = analysis.merged(reads, (tset, d, name), ch, din)
        assert np.array_equal(x["obs"], obs), f"d{d} {name}: observables differ from the test set's"
        notes = analysis.check_net(recs, fz, name, fn["tau"], a.dry)
        cl, clr = analysis.pair(x["c"].astype(np.float32), dec["corr_pm"], fn["tau"])
        assert np.array_equal(clr, x["routed"].astype(bool)) and np.array_equal(cl, x["dec"]), \
            f"d{d} {name}: the saved routing or decisions are not |c| < tau with correlated PM's answers"
        g, grecs = analysis.merged(gs, (tset, d, gapsw.KEY), ch, din)
        for r in grecs:
            assert r["gapsw_sha256"] == own, f"{r['npz']}: made by another gapsw.py"
            assert r["code_digest"] == fz["read_code_digest"] or a.dry, f"{r['npz']}: another read path's code"
        assert np.array_equal(g["obs"], obs), f"d{d}: the front end's observables differ from the test set's"
        assert np.array_equal(g["dec"], dec["plain_pm"]), f"d{d}: the front end's answers are not plain PM's test read's"
        gap = g["gap"]
        routed = gap < pol["tau"]
        gsw = np.where(routed, dec["corr_pm"], g["dec"]).astype(np.uint8)
        fg, fc = gsw != obs, cl != obs
        n = len(obs)
        s = stats.paired(fg, fc, ALPHA)
        tests.append(dict(d=d, convlines=name, gapsw_q=pol["q"], gapsw_tau=pol["tau"], gapsw_failures=s["fail_a"],
                          convlines_failures=s["fail_b"], only_gapsw=s["only_a"], only_convlines=s["only_b"],
                          mcnemar_p=s["mcnemar_p"], diff_ler=s["diff_ler"], lo95=s["diff_lo"], hi95=s["diff_hi"]))
        cq = fn.get("q")
        rows = []
        for t in pol["table"]:
            r = gap < t["tau"]
            rows.append(dict(d=d, q=t["q"], tau_q=t["tau"], routed_test=int(r.sum()), routed_share_test=float(r.mean()),
                             failures=int((np.where(r, dec["corr_pm"], g["dec"]) != obs).sum()),
                             calibration_failures=t["failures"], projected_us=t["projected_us"],
                             gapsw_chosen=bool(abs(t["q"] - pol["q"]) < 1e-12),
                             convlines_q=bool(cq is not None and abs(t["q"] - cq) < 1e-12)))
        curves += rows
        tm = times(a.cpu, d, pol, own, a.dry, din)
        mc = main_res["timing"]["composites"].get(name) if main_res else None
        trows.append(dict(d=d, gapsw_front_us=tm["front_grace_us"], gapsw_routed_us=tm["routed_grace_us"],
                          gapsw_composite_us=tm["composite_us"], gapsw_projected_us=pol["chosen"]["projected_us"],
                          convlines=name, convlines_composite_us=mc["composite_us"] if mc else None,
                          convlines_session_qualified=main_res["timing"]["qualified"] if main_res else None,
                          corr_pm_registered_us=cfg["registered_times_us"][k]["corr_pm"],
                          plain_pm_registered_us=cfg["registered_times_us"][k]["plain_pm"]))
        res["distances"][k] = dict(
            shots=n, policy=dict(q=pol["q"], tau=pol["tau"], operative=pol["operative"], outcome=pol["outcome"],
                                 registered_rule=pol["registered"]["outcome"], sha256=psha),
            convlines=dict(network=name, q=cq, tau=fn["tau"], routed_test=int(clr.sum()), notes=notes,
                           exception=exc.get(d)),
            gapsw_routed_test=int(routed.sum()), gapsw_routed_share_test=float(routed.mean()),
            failures=dict(gapsw=rate(int(fg.sum()), n), convlines=rate(int(fc.sum()), n),
                          corr_pm=rate(int((dec["corr_pm"] != obs).sum()), n),
                          plain_pm=rate(int((dec["plain_pm"] != obs).sum()), n)),
            gapsw_vs_corr_pm=stats.paired(fg, dec["corr_pm"] != obs, ALPHA), times=tm)
    adj, rej = analysis.holm([t["mcnemar_p"] for t in tests], ALPHA)
    for t, p, r in zip(tests, adj, rej):
        t.update(holm_p=p, verdict=("ConvLines pair fewer failures" if t["only_convlines"] < t["only_gapsw"] else
                                    "gap switching fewer failures") if r else "inconclusive")
    res.update(family_G=dict(tests=tests, m=len(tests), note="exact two-sided McNemar per d, Holm over the distances at "
                             "0.05; Newcombe 95 % intervals pointwise"), curves=curves, times=trows)
    os.makedirs(a.out)
    with open(os.path.join(a.out, "gapsw_analysis.json"), "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    analysis.write_csv(os.path.join(a.out, "gapsw_family_G.csv"), tests)
    analysis.write_csv(os.path.join(a.out, "gapsw_curves.csv"), curves)
    analysis.write_csv(os.path.join(a.out, "gapsw_times.csv"), trows)
    f, fmt = res["distances"], analysis.fmt

    def iv(x):
        return f"{x['failures']:,} [{1e2 * x['lo95']:.3f}, {1e2 * x['hi95']:.3f}] %"
    md = [f"# SURF-v2 secondary comparison: gap switching{' (DRY RUN: stand-ins, never a result)' if a.dry else ''}", "",
          f"Freeze commit {res['freeze_commit']}; test set {tset}. Gap switching: plain PM decides, the shots of lowest "
          "complementary gap take correlated PM's answer (surf_v2/gapsw.py, its q from the calibration shots).", "",
          "## Failures on the test set (Wilson 95 % interval of the logical error rate)", "",
          "| d | shots | gap switching (q) | ConvLines pair (network, q) | correlated PM | plain PM |", "|---|---|---|---|---|---|"]
    for d in cfg["distances"]:
        x = f[str(d)]
        md.append(f"| {d} | {x['shots']:,} | {iv(x['failures']['gapsw'])} ({fmt(x['policy']['q'], 2)}) | "
                  f"{iv(x['failures']['convlines'])} ({x['convlines']['network']}, {fmt(x['convlines']['q'], 2)}) | "
                  f"{iv(x['failures']['corr_pm'])} | {iv(x['failures']['plain_pm'])} |")
    md += ["", "## Family G: gap switching against the ConvLines pair (McNemar, Holm over the distances at 0.05)", "",
           "| d | failures gap switching vs ConvLines | only gap switching | only ConvLines | p | Holm p | 95 % interval "
           "of the LER difference | verdict |", "|---|---|---|---|---|---|---|---|"]
    for t in tests:
        md.append(f"| {t['d']} | {t['gapsw_failures']:,} vs {t['convlines_failures']:,} | {t['only_gapsw']:,} | "
                  f"{t['only_convlines']:,} | {t['mcnemar_p']:.3g} | {t['holm_p']:.3g} | [{t['lo95']:.3g}, {t['hi95']:.3g}] | "
                  f"{t['verdict']} |")
    md += ["", "## Composite times (us per shot, Grace basis)", "",
           "| d | gap switching: front end + routed = composite (calibration projection) | ConvLines pair | correlated PM "
           "(registered) | plain PM (registered) |", "|---|---|---|---|---|"]
    for r in trows:
        md.append(f"| {r['d']} | {fmt(r['gapsw_front_us'], 2)} + {fmt(r['gapsw_routed_us'], 2)} = "
                  f"{fmt(r['gapsw_composite_us'], 2)} ({fmt(r['gapsw_projected_us'], 2)}) | {fmt(r['convlines_composite_us'], 2)}"
                  f"{'' if r['convlines_session_qualified'] in (None, True) else ' (session not qualified)'} | "
                  f"{fmt(r['corr_pm_registered_us'], 2)} | {fmt(r['plain_pm_registered_us'], 2)} |")
    md += ["", "## Gap switching against q (tau_q from the calibration shots)", "",
           "| d | q | routed on the test | failures | calibration failures | projected us | mark |", "|---|---|---|---|---|---|---|"]
    for r in curves:
        mark = ", ".join(m for m, on in (("gap switching's q", r["gapsw_chosen"]), ("ConvLines pair's q", r["convlines_q"])) if on)
        md.append(f"| {r['d']} | {fmt(r['q'], 2)} | {r['routed_test']:,} | {r['failures']:,} | {r['calibration_failures']:,} | "
                  f"{fmt(r['projected_us'], 2)} | {mark} |")
    with open(os.path.join(a.out, "gapsw_tables.md"), "x") as fh:
        fh.write("\n".join(md) + "\n")
    for t in tests:
        print(f"d = {t['d']}: gap switching {t['gapsw_failures']} vs ConvLines {t['convlines_failures']} ({t['convlines']}): "
              f"p {t['mcnemar_p']:.3g}, Holm {t['holm_p']:.3g}: {t['verdict']}", flush=True)
    print(f"outputs in {a.out}: " + ", ".join(sorted(os.listdir(a.out))), flush=True)


if __name__ == "__main__":
    main()
