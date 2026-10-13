"""The registered routed-fraction rule (C4.2) exactly as the deployed networks' policies ran it
(ising-arch:ising_compare/arch/policy.py `fit`, lines 117-165), applied to a SURF-v2 network's calibration read.

For each q of the distance's grid (surf_v2/config.json qrule): tau_q = float(np.quantile(|c|, q)) for q > 0 (numpy's
default interpolation on the float32 |c|), tau_0 = 0; a shot is routed when |c| < tau_q, strictly; F_cal(q) counts the
failures when the routed shots take correlated PM's calibration answer and the others the network's (c > 0). The
projection: d = 7 the locked measured complete paths per q (None: not admissible); d = 9-17 projected(0) = net_us and
projected(q) = net_us + router_us + r_q x cost(q), r_q = routed / N the realized fraction and cost(q) the mean Grace
time of correlated PM over the shots routed at q (0 when none is routed). Admissible: projected(q) <= bar. With the
limit L = M - 2 sqrt(2 M): the smallest admissible q with F_cal(q) <= L, else the fallback of the distance's registered
policy (config qrule.fallback; policy.py fit --fallback): at d = 7 the largest admissible q (the deployed d = 7 policy
was fitted before G7 with that rule), at d = 9-17 the admissible q with the fewest calibration failures (ties: the
smaller q); no admissible q: no policy (the network alone, labelled "no admissible q").

    python -m surf_v2.qrule fit --net d11_b5_sv2 --read READS/sv2cal_d11_d11_b5_sv2_0-19.npz
        --calcost CC/calcost_d11.npz --out POL/policy_d11_b5_sv2.json
    python -m surf_v2.qrule reproduce --dir /scratch/alpine/$USER/ising_arch/out/policy --out REPRO.json
reproduce applies the same code to the studies' calibration records (and their frontiers' costs) and must give the
registered choices (config registered_choices: q = 4, 1, 1, 2, 6 and 10 % at d = 7-17), their tau, routed shots,
calibration failures and projected times; it also reads each registered policy record (sha256 checked) and requires its
fallback to be the config's at that distance."""
import argparse
import hashlib
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "config.json")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def config():
    return json.load(open(CONFIG))


def settings(cf, d):
    """The rule's registered inputs at distance d (config qrule): grid, target M, bar, projection and fallback."""
    q, k = cf["qrule"], str(d)
    return dict(grid=q["grid"][k], M=q["M"][k], bar_us=q["bar_us"][k], proj=q["projection"][k], fallback=q["fallback"][k])


def policy_fallback(pol):
    """A registered policy record's fallback: its field, or (fits made before G7 have none) 'largest' by its rule."""
    if "fallback" in pol:
        return pol["fallback"]
    assert "else the largest admissible q" in pol["rule"], f"no fallback in the record's rule: {pol['rule']}"
    return "largest"


def fit_rule(c, corr, obs, grid, M, bar_us, proj, cost=None, fallback="argmin"):
    """policy.py fit's table and choice. c: (N,) float32 logits of the network's calibration read; corr, obs: (N,) uint8
    correlated PM's answers and the observables; proj: config qrule.projection[d] ({"kind": "path", "path_us"} or
    {"kind": "frontier", "net_us", "router_us"}); cost(q, routed) -> cost(q) in Grace us (frontier only); fallback when
    no admissible q meets the limit: "argmin" (the fewest calibration failures, ties to the smaller q) or "largest" (the
    largest admissible q), as policy.py fit --fallback."""
    c = np.asarray(c)
    assert c.dtype == np.float32, "c must be the read's float32 logits"
    az, netd = np.abs(c), (c > 0).astype(np.uint8)
    corr, obs = np.asarray(corr, np.uint8), np.asarray(obs, np.uint8)
    path = {round(float(q), 6): us for q, us in proj["path_us"].items()} if proj["kind"] == "path" else None
    rows = []
    for q in sorted(grid):
        tau = float(np.quantile(az, q)) if q > 0 else 0.0
        r = az < tau
        x = dict(q=q, tau=tau, routed=int(r.sum()), failures=int((np.where(r, corr, netd) != obs).sum()))
        if path is not None:
            x["projected_us"] = path[round(q, 6)]     # None: not measured and above a measured point over the bar
            x["admissible"] = x["projected_us"] is not None and x["projected_us"] <= bar_us
        else:
            x["cost_us"] = (float(cost(q, r)) if r.any() else 0.0) if q > 0 else None
            x["projected_us"] = proj["net_us"] + (proj["router_us"] + float(r.mean()) * x["cost_us"] if q > 0 else 0.0)
            x["admissible"] = x["projected_us"] <= bar_us
        x.update(routed_fraction=float(r.mean()),
                 tau_float32_exact=bool(q == 0 or float(np.float32(tau)) == tau),
                 routing_equal_in_double=bool(np.array_equal(r, np.abs(c.astype(np.float64)) < tau)))
        rows.append(x)
    lim = M - 2 * math.sqrt(2 * M)
    adm = [x for x in rows if x["admissible"]]
    ok = [x for x in adm if x["failures"] <= lim]
    assert fallback in ("argmin", "largest"), fallback
    chosen = (ok[0] if ok else (min(adm, key=lambda x: (x["failures"], x["q"])) if fallback == "argmin" else adm[-1])) \
        if adm else None
    return dict(grid=sorted(grid), margin=M, limit=lim, bar_us=bar_us, projection=proj, table=rows, chosen=chosen,
                qualifies=bool(ok), eligible=bool(adm), fallback=fallback, shots=int(len(c)),
                network_failures=int((netd != obs).sum()), pm_corr_failures=int((corr != obs).sum()),
                outcome=("q = 0: the network alone" if chosen and chosen["q"] == 0 else "routed pair" if chosen else
                         "no admissible q: the network alone, labelled"))


def load_read(path):
    """A calibration read of convlines.evaluate --save (npz and its .json record), checked against the record."""
    rec = json.load(open(path[:-4] + ".json"))
    assert rec["npz_sha256"] == sha_file(path), f"{path}: not the file its record names"
    z = np.load(path)
    for k, v in rec["arrays"].items():
        assert hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() == v["sha256"], f"{path}: {k} differs"
    return z, rec


def load_calcost(path):
    rec = json.load(open(path[:-4] + ".json"))
    assert rec["npz_sha256"] == sha_file(path), f"{path}: not the file its record names"
    return np.load(path), rec


def fit(a):
    cf = config()
    z, rec = load_read(a.read)
    cc, crec = load_calcost(a.calcost)
    d = rec["d"]
    want = (cf["dry"]["set"], cf["dry"]["cal_chunks"]) if a.dry else (cf["calibration"]["set"], cf["calibration"]["chunks"])
    assert (rec["set"], rec["chunks"]) == want, f"not the calibration set {want}"
    assert rec["decoder"] == a.net and rec["network"] == a.net and rec["tau"] is None, \
        "the read must be the network alone (--tau none) of --net"
    assert rec["routing"] == "deployed" or a.dry, "tau is fitted on the deployed engine's c (--routing deployed)"
    assert crec["d"] == d and crec["manifest_sha256"] == rec["manifest_sha256"], "calcost belongs to another set"
    assert np.array_equal(cc["obs"], z["obs"]), "calcost's shots are not the read's"
    t_grace = cc["time_us"] * crec["ratio"]
    pol = fit_rule(z["c"], cc["dec"], z["obs"], cost=lambda q, r: t_grace[r].mean(), **settings(cf, d))
    pol.update(rule=cf["qrule"]["source"], d=d, network=a.net, network_sha256=rec["network_sha256"],
               read=os.path.basename(a.read), read_sha256=sha_file(a.read), read_record_sha256=sha_file(a.read[:-4] + ".json"),
               read_code_digest=rec["code_digest"], read_commit=rec["release_commit"], calcost=os.path.basename(a.calcost),
               calcost_sha256=sha_file(a.calcost), calcost_ratio=crec["ratio"], config_sha256=sha_file(CONFIG))
    if pol["chosen"] is not None:
        assert pol["chosen"]["tau_float32_exact"] and pol["chosen"]["routing_equal_in_double"], \
            "tau routes differently in float32 and in double: stop and report"
    for x in pol["table"]:
        print(f"q={100 * x['q']:4.1f} % tau={x['tau']:.6f} routed={x['routed']:6d} F_cal={x['failures']} projected "
              f"{x['projected_us']} admissible {x['admissible']}", flush=True)
    print(f"{a.net}: limit {pol['limit']:.1f}; chosen {pol['chosen']}; {pol['outcome']}", flush=True)
    with open(a.out, "x") as fh:
        json.dump(pol, fh, indent=1)


def reproduce(a):
    """The rule on the studies' calibration records -> their registered choices (config registered_choices)."""
    cf = config()
    reg = cf["registered_choices"]
    out, ok = {}, True
    for k in ("7", "9", "11", "13", "15", "17"):
        r = reg[k]
        f = os.path.join(a.dir, r["calib"])
        assert sha_file(f) == r["calib_sha256"], f"{f}: not the registered calibration record"
        z = np.load(f)
        c = z["z"]
        cost = None
        if "frontier" in r:
            fr = os.path.join(a.dir, r["frontier"])
            assert sha_file(fr) == r["frontier_sha256"], f"{fr}: not the registered frontier"
            fj = json.load(open(fr))
            assert fj["calib_sha256"] == r["calib_sha256"]
            tab = {round(x["q"], 6): x["cond_cost_us_grace"] for x in fj["rows"]}
            cost = lambda q, routed, tab=tab: tab[round(q, 6)]          # noqa: E731  the studies' measured cost(q)
        pf = os.path.join(a.dir, r["policy_file"])
        assert sha_file(pf) == r["policy_sha256"], f"{pf}: not the registered policy"
        rp = json.load(open(pf))
        assert rp["calib_sha256"] == r["calib_sha256"], f"{pf}: another calibration record's policy"
        pol = fit_rule(c, z["pm_corr"], z["obs"], cost=cost, **settings(cf, int(k)))
        ch = pol["chosen"]
        got = dict(q=ch["q"], tau=ch["tau"], routed=ch["routed"], failures=ch["failures"],
                   projected_us=ch["projected_us"], qualifies=pol["qualifies"], fallback=pol["fallback"])
        want = dict({x: r[x] for x in got if x != "fallback"}, fallback=policy_fallback(rp))
        same = {x: (got[x] == want[x] if x != "projected_us" else abs(got[x] - want[x]) <= 1e-9 * max(1, abs(want[x])))
                for x in got}
        ok &= all(same.values())
        out[k] = dict(got=got, registered=want, equal=same, all_equal=all(same.values()), table=pol["table"],
                      calib_sha256=r["calib_sha256"], frontier_sha256=r.get("frontier_sha256"))
        print(f"d = {k}: chosen q {100 * got['q']:g} % tau {got['tau']!r} routed {got['routed']} F_cal {got['failures']} "
              f"projected {got['projected_us']}; registered q {100 * want['q']:g} % tau {want['tau']!r}: "
              f"{'EQUAL' if all(same.values()) else 'DIFFERS ' + str(same)}", flush=True)
    res = dict(what="surf_v2.qrule reproduce: the registered choices from the studies' calibration records",
               numpy=np.__version__, all_equal=ok, d=out)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)
    print(f"reproduction: {'ALL EQUAL' if ok else 'NOT EQUAL'}", flush=True)
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--net", required=True)
    p.add_argument("--read", required=True, help="the network's calibration read (convlines.evaluate --save .npz)")
    p.add_argument("--calcost", required=True, help="surf_v2.calcost's output of the same calibration set")
    p.add_argument("--dry", action="store_true", help="dry runs only: the dry set, and c of the fp32 reference or a stand-in")
    p.add_argument("--out", required=True)
    p = sub.add_parser("reproduce")
    p.add_argument("--dir", required=True, help="the studies' policy directory (calibration records and frontiers)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"fit": fit, "reproduce": reproduce}[a.cmd](a)


if __name__ == "__main__":
    main()
