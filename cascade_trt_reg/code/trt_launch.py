"""CASCADE-TRT step 4 (exploratory): launcher of the re-test harnesses' pipeline subcommands on the public tree, with the
Cascade network on the registered compiled path (CTRT_ENGINE=compiled) or on a TensorRT plan (CTRT_ENGINE=trt).

    python trt_launch.py SCRIPT ARGS...    SCRIPT = .../phase_d/d1_m2.py or .../phase_c/pc_speed.py (BB72),
                                           or surface.sv_speed (d = 13), followed by the harness's own arguments

Nothing inside a timed region changes. Patches, all recorded in every row under "ctrt":
  seeds    kind 12 (the speed stream) -> the registered H100 run's beacon seeds (speed_rt/cascade/seeds.json), at the top
           level so that every spawned sampling worker patches its own copy (as speed_rt/cascade/srt_cas.py does)
  card     pc_common.GPU / sv_common.GPU = the H100's name (srt_cas.py's admit_h100, srt_d13.py)
  weights  pc_common.NETS -> the public files; bb.cascade_eval.sha256_file -> the source_sha256 each public (stripped)
           file carries, which the locks name (same tensors)
  d1       m2.frozen_speed -> the public speed components and D1's lock; m2.read_ref -> a one-row dummy: C4b's reads are
           not public, so the rows' S_check is void
  d13      sv_common.same_identity -> gpu and torch only; sv_common.kcache -> the lock's digest (the pod's own caches)
  engine   trt: pc_speed.build(pr, 'compiled', B) of a Cascade network -> trt_common.TRTGraphBB on the B plan;
           sv_common.Graph of B's function -> trt_common.TRTGraphD13 on the B plan"""
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ENGINE = os.environ.get("CTRT_ENGINE", "compiled")
PUB = os.environ["CTRT_PUB"]
PLANS = os.environ.get("CTRT_PLANS", "")
H100 = "NVIDIA H100 80GB HBM3"
SEEDS = json.load(open(os.path.join(PUB, "speed_rt", "cascade", "seeds.json")))["seeds"]
BB_SETS = {("depspam_fm", 2000): "srt-cas-dfm-p2", ("depspam_fm", 3000): "srt-cas-dfm-p3",
           ("depspam_fm", 5500): "srt-cas-dfm-p55", ("uniform", 2000): "srt-cas-uni-p2",
           ("uniform", 3000): "srt-cas-uni-p3", ("uniform", 5500): "srt-cas-uni-p55"}
D13_SETS = {6000: "srt-cas-d13-p6", 3500: "srt-cas-d13-p35"}
INFO = {"engine": ENGINE, "seeds": "registered H100 kind-12 beacon seeds (seeds.json)", "trt": {}, "notes": []}
if ENGINE not in ("compiled", "trt"):
    raise SystemExit(f"CTRT_ENGINE={ENGINE!r}")


def install_seeds():
    from bb import cascade_data as cd
    if not getattr(cd.eval_seed, "_ctrt", False):
        o_bb = cd.eval_seed

        def eval_seed(kind, p, chunk, family="uniform"):
            base = o_bb(kind, p, chunk, family)
            if kind != 12:
                return base
            return SEEDS[BB_SETS[(family, int(round(p * 1e6)))]][chunk]
        eval_seed._ctrt = True
        cd.eval_seed = eval_seed
    try:
        from surface import surface_data as sd
    except ImportError:
        return
    if not getattr(sd.eval_seed, "_ctrt", False):
        o_sd = sd.eval_seed

        def eval_seed13(d, kind, p, chunk):
            base = o_sd(d, kind, p, chunk)
            if kind != 12:
                return base
            if d != 13:
                raise ValueError(f"ctrt: kind 12 at d = {d}")
            return SEEDS[D13_SETS[int(round(p * 1e6))]][int(chunk)]
        eval_seed13._ctrt = True
        sd.eval_seed = eval_seed13


install_seeds()                                  # top level: the main process and every spawned worker


def wrap_emit(mod):
    o = mod.emit

    def emit(path, row):
        return o(path, {**row, "ctrt": INFO})
    mod.emit = emit


def patch_bb(stem, mod):
    import torch
    import pc_common as pc
    import pc_speed as ps
    from bb import cascade_eval as ce
    pc.GPU = H100
    W = os.path.join(PUB, "weights", "cascade_bb72")
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
        sp = os.path.join(PUB, "cascade", "bb72", "cascade_compare", "phase_c", "out", "speed_components.json")
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


def patch_d13(mod):
    from surface import sv_common as sc
    sc.GPU = H100
    lock = None
    if "--lock" in sys.argv:
        lock = json.load(open(sys.argv[sys.argv.index("--lock") + 1]))
    o_same = sc.same_identity

    def same_identity(rec, ident, what, *args, **kw):
        INFO["identity_checked"] = ["gpu", "torch"]
        for k in ("gpu", "torch"):
            if rec.get(k) != ident.get(k) and k in ident:
                raise SystemExit(f"ctrt: {what}: {k} {rec.get(k)!r} is not the lock's {ident.get(k)!r}")
    sc.same_identity = same_identity
    if lock is not None:
        o_kc = sc.kcache

        def kcache():
            INFO["kcache_pod"] = o_kc()
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


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    script = argv[0]
    if script.endswith(".py"):
        script = os.path.abspath(script)
        stem = os.path.basename(script)[:-3]
        if stem not in ("d1_m2", "pc_speed"):
            raise SystemExit(f"ctrt: {stem}.py is not a harness of this launcher")
        sys.argv = [script] + list(argv[1:])
        sys.path.insert(0, os.path.dirname(script))
        mod = importlib.import_module(stem)
        patch_bb(stem, mod)
        (mod.m2.main if stem == "d1_m2" else mod.main)()
    elif script == "surface.sv_speed":
        sys.argv = [script] + list(argv[1:])
        mod = importlib.import_module(script)
        patch_d13(mod)
        mod.main()
    else:
        raise SystemExit(f"ctrt: unknown harness {script!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
