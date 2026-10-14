"""CASCADE-TRT-REG's run code, written after F and published with B (before any timing): what the launchers ctrt_cas.py,
ctrt_d13.py and ctrt_check.py and the driver ctrt_drive.py share. Importing it changes three things and nothing else:

- the timing shots: stage 1's srtlib (speed_rt/cascade/srtlib.py) runs unchanged, with this test's sets in place of stage
  1's: set_defs() reads cascade_trt_reg/sets_ctrt.json (stage 1's kinds, p and chunking under new names, so new tags and
  seeds), where srtlib's own reads sets_cas.json. The seeds file and the beacon record come from SRT_SEEDS and SRT_BEACON
  and are checked by srtlib against each other, every seed recomputed, as in stage 1;
- the Cascade engine: CTRT_ENGINE = compiled (the registered path, untouched) or trt (code/trt_common.py's TensorRT plan in
  a CUDA graph behind the harness's graph interface, the plans in CTRT_PLANS);
- the public tree: the patches of code/trt_launch.py that let the registered harnesses run on the public tree (the public
  weights and the source_sha256 they carry, D1's speed components, C4b's reads, at d = 13 the identity reduced to GPU and
  torch and the lock's cache digest). trt_launch.py's seeds (stage 1's H100 seeds) and its card line are not used: the
  seeds are srtlib's and the card is admitted by stage 1's launchers (srt_cas.py, srt_d13.py) from SRT_CARD."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.dirname(HERE)                                   # cascade_trt_reg/
ROOT = os.path.dirname(REG)                                   # the public tree
CAS = os.path.join(ROOT, "speed_rt", "cascade")
CODE = os.path.join(REG, "code")
for _p in (CODE, CAS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import srtlib  # noqa: E402

SETS = os.path.join(REG, "sets_ctrt.json")
ENGINE = os.environ.get("CTRT_ENGINE", "compiled")
if ENGINE not in ("compiled", "trt"):
    raise SystemExit(f"ctrt: CTRT_ENGINE={ENGINE!r}, not compiled or trt")
PLANS = os.environ.get("CTRT_PLANS", "")
INFO = {"engine": ENGINE, "card": os.environ.get("SRT_CARD", "rtx").lower(), "trt": {}}


def set_defs():
    """{name: entry} of this test's sets, in place of srtlib.set_defs (stage 1's sets_cas.json)."""
    return json.load(open(SETS))["sets"]


srtlib.set_defs = set_defs


def wrap_emit(mod):
    """Every row the harness writes carries INFO under "ctrt" (engine, card, plans, patches)."""
    o = mod.emit

    def emit(path, row):
        return o(path, {**row, "ctrt": INFO})
    mod.emit = emit


def patch_bb(stem, mod):
    """code/trt_launch.py's patch_bb without its card line, after the harness is imported."""
    import torch
    import pc_common as pc
    import pc_speed as ps
    from bb import cascade_eval as ce
    W = os.path.join(ROOT, "weights", "cascade_bb72")
    for t in list(pc.NETS):
        f = os.path.join(W, f"{t}.pt")
        if os.path.exists(f):
            pc.NETS[t] = (f, *pc.NETS[t][1:])
    o_sha = ce.sha256_file

    def sha256_file(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        s = ck.get("source_sha256") if isinstance(ck, dict) else None
        INFO.setdefault("sha_source", {})[os.path.basename(path)] = bool(s)
        return s or o_sha(path)
    ce.sha256_file = sha256_file
    wrap_emit(ps)
    if stem == "d1_m2":
        m2 = mod.m2
        sp = os.path.join(ROOT, "cascade", "bb72", "cascade_compare", "phase_c", "out", "speed_components.json")
        lock = mod.LOCK

        def frozen_speed():
            ok = pc.sha256_file(sp) == json.load(open(lock))["speed_sha256"]
            INFO["speed_components_sha_matches_lock"] = ok
            return json.load(open(sp)), {"depspam_fm": lock}
        m2.frozen_speed = frozen_speed

        def read_ref(t, fam, p, dev):
            INFO["s_check"] = "void: C4b's reads are not public (one dummy row)"
            return (torch.zeros(1, 432, dtype=torch.uint8, device=dev), torch.zeros(1, pc.K, dtype=torch.uint8, device=dev),
                    torch.zeros(1, dtype=torch.float32, device=dev))
        m2.read_ref = read_ref
    if ENGINE == "trt":
        import trt_common as tc
        o_build = ps.build

        def build(pr, path, B, score="minabs"):
            if pr.kind == "cascade" and path == "compiled":
                if score != "minabs":
                    raise SystemExit(f"ctrt: the TensorRT plan computes the minabs confidence, not {score!r}")
                g = tc.TRTGraphBB(tc.plan_path(PLANS, "bb", B), B, 432, torch.device("cuda"))
                INFO["trt"][str(B)] = g.info()
                return g
            return o_build(pr, path, B, score)
        ps.build = build


def patch_d13(mod, argv):
    """code/trt_launch.py's patch_d13 without its card line (srt_d13.py admits the card), before sv_speed.main runs."""
    from surface import sv_common as sc
    lock = json.load(open(argv[argv.index("--lock") + 1])) if "--lock" in argv else None

    def same_identity(rec, ident, what, *args, **kw):
        INFO["identity_checked"] = ["gpu", "torch"]
        for k in ("gpu", "torch"):
            if rec.get(k) != ident.get(k) and k in ident:
                raise SystemExit(f"ctrt: {what}: {k} {rec.get(k)!r} is not the lock's {ident.get(k)!r}")
    sc.same_identity = same_identity
    if lock is not None:
        o_kc = sc.kcache

        def kcache():
            INFO["kcache_here"] = o_kc()
            return dict(lock["kcache"])
        sc.kcache = kcache
    wrap_emit(mod)
    if ENGINE == "trt":
        import torch
        import trt_common as tc
        o_b_fn, o_graph = sc.b_fn, sc.Graph

        class BFn:
            def __call__(self, x):
                raise RuntimeError("ctrt: B runs on its TensorRT plan")

        def b_fn(model, path):
            o_b_fn(model, path)                          # the same refusal of an unknown path
            return BFn()

        def Graph(fn, B, Dd, dev, calls=1):
            if isinstance(fn, BFn):
                if calls != 1:
                    raise SystemExit("ctrt: TensorRT B at calls = 1 only")
                g = tc.TRTGraphD13(tc.plan_path(PLANS, "d13", B), B, Dd, torch.device("cuda"))
                INFO["trt"][str(B)] = g.info()
                return g
            return o_graph(fn, B, Dd, dev, calls)
        sc.b_fn, sc.Graph = b_fn, Graph
