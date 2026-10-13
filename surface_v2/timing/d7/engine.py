"""The d = 7 executable's driver: the resident kernel (kernel.cu) and the compiled router (router.cc), both built by
build.sh, through ctypes.

  pack(tables)   the kernel's buffers from convlines.net.deploy_tables (engine site order, fp32 on the GPU): fp16 weight
                 rows padded to the kernel's shared-memory strides, fp32 small parameters, per-CTA neighbour tables
  Engine         Engine(tables, lib, ctx, cubin).start() launches the resident 7-CTA cluster on its own non-blocking
                 stream and waits until it reports ready (weights and tables in shared memory);
                 lib.run_persistent(eng.pers, ...) serves shots through the mapped request line (router.cc
                 s7_run_persistent's loop); stop() sends the stop request, waits for the kernel's final state and
                 synchronises its stream. While the kernel is resident nothing may synchronise the device
                 (torch.cuda.synchronize, frees, other engines): it ends only on the stop request (or its watchdog).
                   body 1  the network (the studies' executable, with fwd = 1)
                   body 0  handshake microbenchmark: detect in every CTA, one cluster barrier, CTA 0 publishes 0
                   body 2  handshake microbenchmark: CTA 0 publishes on its own detection (no cluster barrier)
  Lib            the router library libs7router.so: correlated PyMatching, the persistent mode's calls and the
                 graph mode (s7_run, s7_tail_launch) through which timing.gpu_term measures the handoff at
                 d = 9-17

The code is the studies' (only comments, the cubin's location and the router wrapper's development options, which are
left out, differ); the kernel's identifiers keep their development names (s9_*, s7_*)."""
import ctypes
import os
import time

import numpy as np
import torch

from timing.d7 import cudrv

NC, RC, RP, KSP, KB, NB, NS, NTH = 7, 48, 48, 80, 27, 32, 32, 384
D, M, C, CB, FF, NX = 7, 336, 64, 16, 128, 192
ZB = ZR = 336
SWB, SWD, SWC, SWU, SWS, SWV = 88, 72, 24, 24, 72, 72                    # shared row strides (halves)
T0B = RP * KSP * 2 + 128 + RP * C * 4
T1B = RP * NB * 2 + RP * NS * 4 + RP * 8 * 4
XP = D * C + 16
FLOATS = ("b2", "b0", "inv_n0", "eps_b0", "eps_b1", "eps_h", "eps_m")


def fl_offsets(L):
    """Fl<L> (kernel.cu): offsets (floats) of the packed small parameters."""
    o = dict(LNB=0)
    o["BD"] = o["LNB"] + L * 2 * C
    o["BC"] = o["BD"] + L * CB
    o["BU"] = o["BC"] + L * 2 * CB
    o["LNH"] = o["BU"] + L * C
    o["BV"] = o["LNH"] + 2 * C
    o["U0"] = o["BV"] + C
    o["BO"] = o["U0"] + D * C
    o["LNM"] = o["BO"] + C
    o["B1"] = o["LNM"] + 2 * C
    o["W2"] = o["B1"] + FF
    o["WL"] = o["W2"] + FF
    o["N"] = o["WL"] + 8
    return o


def smem_bytes(L):
    """The shared-memory bytes of kernel.cu's Lay<L> without its 64-byte forwarding slot."""
    A0 = 352 + T0B + T1B
    Wb = A0 + RP * 88 * 2
    Wd = Wb + C * 88 * 2
    Wc = Wd + L * CB * 72 * 2
    Wu = Wc + L * KB * CB * 24 * 2
    Wsc = Wu + L * C * 24 * 2
    Wv = Wsc + NS * 72 * 2
    u = Wv + C * 72 * 2
    Y = u + L * (M + 1) * CB * 2
    V = Y + 3 * 16 * 72 * 2
    cp = V + 3 * 16 * 24 * 2
    red = cp + 3 * 4 * 256 * 4
    G16 = red + 2 * 3 * 4 * 16 * 4
    V16 = G16 + 3 * 16 * 40 * 2
    pp = V16 + 3 * 16 * 72 * 2
    part = pp + 3 * 28 * 16 * 4
    gs = part + NC * C * 4
    zj = gs + NC * 8 * 4
    zbar = zj + 8 * 4
    Wo = zbar + 32
    W1 = Wo + C * C * 2
    fl = W1 + C * FF * 2
    ra = fl + fl_offsets(L)["N"] * 4
    return ra + (C + 4 * C + C + 3 * FF + 16) * 4


@torch.no_grad()
def pack(t):
    """Device buffers of the kernel's layouts from convlines.net.deploy_tables (engine site order, fp32 on the GPU)."""
    cfg, h = t["cfg"], t["head"]
    L = cfg["blocks"]
    assert (cfg["d"], cfg["m"], cfg["width"], cfg["bott"], t["nx"], h["H"], h["dh"], h["W1"].shape[1]) == \
        (D, M, C, CB, NX, 4, 16, FF) and L in (1, 2), "the d = 7 kernel: the widths-64 / 16 ConvLines only"
    dev = t["P"].device
    f16, f32, i16 = torch.float16, torch.float32, torch.int16
    det = t["det"].long()
    nbs, nbk = t["nb_stem"].long(), t["nb_blk"].long()
    assert nbs.shape == (M, 75) and nbk.shape == (M, KB)
    tapdet = torch.where(nbs >= 0, det[nbs.clamp(min=0)], torch.full_like(nbs, ZB))
    blk = torch.where(nbk >= 0, nbk, torch.full_like(nbk, ZR))
    nbd = torch.full((NC, RP, KSP), ZB, dtype=i16, device=dev)
    nbb = torch.full((NC, RP, NB), ZR, dtype=i16, device=dev)
    own = torch.full((NC, 64), ZB, dtype=i16, device=dev)
    P = torch.zeros(NC, RP, C, dtype=f32, device=dev)
    BIAS = torch.zeros(NC, RP, NS, dtype=f32, device=dev)
    Apar = torch.zeros(NC, RP, 8, dtype=f32, device=dev)
    Bt = h["BIAS"].view(M, D, 4)                                  # [site][line][head]
    assert NX % RC == 0                                           # every CTA holds sites of one type
    for r in range(NC):
        s = slice(r * RC, (r + 1) * RC)
        nbd[r, :RC, :75] = tapdet[s].to(i16)
        nbb[r, :RC, :KB] = blk[s].to(i16)
        own[r, :RC] = det[s].to(i16)
        P[r, :RC] = t["P"][s].float()
        for hh in range(4):
            BIAS[r, :RC, hh * 8:hh * 8 + D] = Bt[s, :, hh].float()
        Apar[r, :RC, :D] = t["A"][s].float()
    Wb = torch.zeros(2, C, KSP, dtype=f16, device=dev)
    Wb[:, :, :75] = t["Wb"].transpose(1, 2).to(f16)               # (2, taps, C) -> [type][n][k]
    bl = t["blocks"]
    Wd = torch.stack([p["Wd"].t() for p in bl]).to(f16)           # (C, CB) -> [n][k]
    Wc = torch.stack([p["Wc"].view(2, KB, CB, CB).permute(0, 1, 3, 2) for p in bl]).to(f16)   # [l][type][tap][n][k]
    Wu = torch.stack([p["Wu"].t() for p in bl]).to(f16)           # (CB, C) -> [n][k]
    o = fl_offsets(L)
    fl = torch.zeros(o["N"], dtype=f32, device=dev)
    for l, p in enumerate(bl):
        fl[o["LNB"] + l * 2 * C:o["LNB"] + l * 2 * C + C] = p["ln_w"].float()
        fl[o["LNB"] + l * 2 * C + C:o["LNB"] + (l + 1) * 2 * C] = p["ln_b"].float()
        fl[o["BD"] + l * CB:o["BD"] + (l + 1) * CB] = p["bd"].float()
        fl[o["BC"] + l * 2 * CB:o["BC"] + (l + 1) * 2 * CB] = p["bc"].float().reshape(-1)
        fl[o["BU"] + l * C:o["BU"] + (l + 1) * C] = p["bu"].float()
    fl[o["LNH"]:o["LNH"] + C] = h["ln_w"].float()
    fl[o["LNH"] + C:o["LNH"] + 2 * C] = h["ln_b"].float()
    fl[o["BV"]:o["BV"] + C] = h["bv"].float()
    fl[o["U0"]:o["U0"] + D * C] = h["U0"][:D].float().reshape(-1)
    fl[o["BO"]:o["BO"] + C] = h["bo"].float()
    fl[o["LNM"]:o["LNM"] + C] = h["m_ln_w"].float()
    fl[o["LNM"] + C:o["LNM"] + 2 * C] = h["m_ln_b"].float()
    fl[o["B1"]:o["B1"] + FF] = h["b1"].float()
    fl[o["W2"]:o["W2"] + FF] = h["W2"].reshape(-1).float()
    fl[o["WL"]:o["WL"] + D] = t["w"].float()
    Wsc = torch.zeros(NS, C, dtype=f32, device=dev)
    for hh in range(4):
        for j in range(D):
            Wsc[hh * 8 + j] = h["Wsc"][:, j * 4 + hh]
    def rows(w, stride):                                         # [..., rows, k] fp16 -> rows padded to the stride
        out = torch.zeros(*w.shape[:-1], stride, dtype=f16, device=dev)
        out[..., :w.shape[-1]] = w.to(f16)
        return out.contiguous()

    def raw(*ts):                                                 # per CTA: the tensors' bytes, concatenated
        return torch.cat([x.contiguous().view(torch.uint8).reshape(NC, -1) for x in ts], 1).contiguous()

    T0 = raw(nbd, own, P)
    T1 = raw(nbb, BIAS, Apar)
    assert T0.shape[1] == T0B and T1.shape[1] == T1B
    buf = dict(T0=T0, T1=T1, Wb=rows(Wb, SWB), Wd=rows(Wd, SWD), Wc=rows(Wc.reshape(L, 2, KB * CB, CB), SWC),
               Wu=rows(Wu, SWU), Wsc=rows(Wsc, SWS), Wv=rows(h["Wv"].t(), SWV), Wo=h["Wo"].to(f16).contiguous(),
               W1=h["W1"].to(f16).contiguous(), fl=fl.contiguous())
    scal = dict(b2=float(h["b2"]), b0=float(t["b0"]), inv_n0=1.0 / float(h["n0"]), eps_b0=float(bl[0]["eps"]),
                eps_b1=float(bl[-1]["eps"]), eps_h=float(h["eps"]), eps_m=float(h["m_eps"]))
    return buf, scal, L



# ---- the resident kernel -----------------------------------------------------------------------------------------------
NT, NDBG = 416, 64
PTRS = ("req", "res", "status", "zout", "T0", "T1", "Wb", "Wd", "Wc", "Wu", "Wsc", "Wv", "Wo", "W1", "fl", "dbg")
OFF = dict(req=0, res=1024, status=1088, zout=1152)   # router.cc s9_pers_new's block
STATES = {0: "empty", 1: "ready", 2: "stopped", 3: "watchdog"}
KERNELS = {1: "s9_serve_l{L}", 0: "s9_hs_cluster", 2: "s9_hs_cta0"}
_resident = []                                         # at most one resident kernel per process


class Params(ctypes.Structure):
    _fields_ = ([(n, ctypes.c_void_p) for n in PTRS] + [(n, ctypes.c_float) for n in FLOATS] +
                [("seq0", ctypes.c_uint32), ("tag0", ctypes.c_uint32), ("nl", ctypes.c_int), ("npoll", ctypes.c_int),
                 ("gap", ctypes.c_int), ("classic", ctypes.c_int), ("check", ctypes.c_int), ("fwd", ctypes.c_int), ("relax", ctypes.c_int),
                 ("watchdog_ns", ctypes.c_longlong), ("xdev", ctypes.c_void_p), ("tau", ctypes.c_double)])


class Engine:
    def __init__(self, t, lib, ctx, cubin, body=1, nl=1, npoll=1, gap=0, classic=0, check=0, fwd=0, relax=1, dbg=False,
                 zout=False, xdev=None):
        cc = torch.cuda.get_device_capability()
        assert cc >= (9, 0) or body != 1, "the engine needs thread-block clusters (sm_90 and later)"
        assert npoll in (1, 2, 4) and nl >= 1 and (classic or npoll <= nl), (npoll, nl)
        self.buf, self.scal, self.L = pack(t)
        self.lib, self.ctx = lib, ctx
        self.cubin = cubin                             # build.sh's kernel_sm90a.cubin
        self.mod = cudrv.Module(open(self.cubin, "rb").read())
        self.smem = self._smem_sizes()[self.L - 1]
        if cc >= (9, 0):                               # the launch kernel's layout and the 64-byte forwarding slot
            assert self.smem == smem_bytes(self.L) + 64, (self.smem, smem_bytes(self.L))
        name = KERNELS[body].format(L=self.L)
        self.kern = self.mod.kernel(name, smem=self.smem)
        self.pers, self.dev = lib.pers_new(ctx, nl)
        self.stream = lib.stream_new(ctx)
        self.dbgbuf = torch.zeros(NDBG, NC, 32, dtype=torch.int64, device="cuda") if dbg else None
        self.zout = zout
        self.xdev = xdev                               # development option (a second-tier decoder's static input)
        assert xdev is None or (xdev.dtype == torch.uint8 and xdev.is_cuda and xdev.numel() == 336 and
                                xdev.data_ptr() % 16 == 0)
        relax = 0 if zout else relax                   # the check buffer is ordered before the word by the release
        self.cfg = dict(kernel=name, body=body, nl=nl, npoll=npoll, gap=gap, classic=classic, check=check, fwd=fwd,
                        relax=relax,
                        smem=self.smem,
                        regs=self.kern.regs(), threads=NT, ctas=NC, cubin=os.path.basename(self.cubin), dbg=bool(dbg),
                        zout=bool(zout))
        self.running = False
        torch.cuda.synchronize()

    def _smem_sizes(self):
        out = torch.zeros(2, dtype=torch.int32, device="cuda")
        self.mod.kernel("s9_smem_sizes")(1, 1, 0, [ctypes.c_void_p(out.data_ptr())])
        torch.cuda.synchronize()
        return [int(v) for v in out.cpu()]

    def _params(self, seq0, tag0, watchdog_ns, tau=0.0, stage=False):
        prm = Params()
        for k, o in OFF.items():
            setattr(prm, k, self.dev + o)
        if not self.zout:
            prm.zout = 0
        for k in PTRS[4:-1]:
            setattr(prm, k, self.buf[k].data_ptr())
        prm.dbg = self.dbgbuf.data_ptr() if self.dbgbuf is not None else 0
        for k in FLOATS:
            setattr(prm, k, self.scal[k])
        prm.seq0, prm.tag0 = seq0, tag0
        prm.nl, prm.npoll, prm.gap, prm.classic = self.cfg["nl"], self.cfg["npoll"], self.cfg["gap"], self.cfg["classic"]
        prm.check, prm.fwd, prm.relax = self.cfg["check"], self.cfg["fwd"], self.cfg["relax"]
        prm.watchdog_ns = int(watchdog_ns)
        prm.xdev = self.xdev.data_ptr() if (stage and self.xdev is not None) else 0
        prm.tau = float(tau)
        return prm

    def start(self, seq0=0, tag0=0, watchdog_ns=0, timeout=10.0, tau=0.0, stage=False):
        """Launch the resident kernel; -> the startup times (ns): the launch call, launch to ready. stage (development
        option): shots with |double(c)| < tau are copied into xdev before a released result word."""
        assert not _resident, "another resident kernel runs in this process"
        torch.cuda.synchronize()                       # every packed buffer is written before the kernel starts
        self.lib.L.s9_pers_reset(self.pers, seq0, tag0)
        self.prm = self._params(seq0, tag0, watchdog_ns, tau, stage)
        self.cfg["staging_tau"] = float(tau) if stage else None
        t0 = time.perf_counter_ns()
        self.kern(NC, NT, self.smem, self.prm, stream=self.stream)
        t1 = time.perf_counter_ns()
        ns = self.lib.L.s9_pers_wait(self.pers, 1 << 1, timeout)
        t2 = time.perf_counter_ns()
        if ns < 0:
            raise RuntimeError(f"the d = 7 kernel: not ready within {timeout} s (status {self.lib.pers_status(self.pers)[:5]})")
        self.running = True
        _resident.append(self)
        return dict(launch_call_ns=t1 - t0, launch_to_ready_ns=t2 - t0)

    def _ended(self, timeout):
        if self.lib.stream_sync(self.ctx, self.stream, timeout) == 2:
            # a CTA still runs: nothing in this process may touch the device again; exit without the CUDA teardown
            print(f"the d = 7 kernel has not ended {timeout} s after its final state "
                  f"(status {self.lib.pers_status(self.pers)[:5]}); exiting", flush=True)
            os._exit(4)

    def stop(self, timeout=5.0):
        """The stop request -> the kernel's final status; its stream done."""
        served = self.lib.pers_stop(self.pers, timeout)
        self._ended(timeout)
        st = self.lib.pers_status(self.pers)
        self.running = False
        if self in _resident:
            _resident.remove(self)
        return dict(served=served, state=STATES.get(st[0], st[0]), seq=st[2], tag=st[3], classic_lag=st[4])

    def stamps(self):
        """The stamp ring (after stop): (NDBG, NC, 32) int64 globaltimer ns, 0 where not written."""
        assert not self.running and self.dbgbuf is not None
        return self.dbgbuf.cpu().numpy()


# ---- the router library ------------------------------------------------------------------------------------------------
class Lib:
    """libs7router.so through ctypes: correlated PyMatching, the calls of the persistent mode (d = 7) and the graph
    mode (the deployed path of the Triton engine at d = 9-17)."""

    def __init__(self, path):
        L = ctypes.CDLL(path)
        vp, i64, u64, cp, ci = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64, ctypes.c_char_p, ctypes.c_int
        u32, dbl = ctypes.c_uint32, ctypes.c_double
        L.s7_pm_new.restype, L.s7_pm_new.argtypes = vp, [cp, i64, cp, ci]
        L.s7_pm_decode_rows.restype, L.s7_pm_decode_rows.argtypes = ci, [vp, vp, i64, vp]
        L.s7_ctx_new.restype, L.s7_ctx_new.argtypes = vp, [cp, ci]
        L.s9_pers_new.restype, L.s9_pers_new.argtypes = vp, [vp, ci, ctypes.POINTER(u64), cp, ci]
        L.s9_pers_reset.restype, L.s9_pers_reset.argtypes = None, [vp, u32, u32]
        L.s9_pers_status.restype, L.s9_pers_status.argtypes = u32, [vp, ci]
        L.s9_pers_wait.restype, L.s9_pers_wait.argtypes = i64, [vp, u32, dbl]
        L.s9_pers_stop.restype, L.s9_pers_stop.argtypes = i64, [vp, dbl, cp, ci]
        L.s9_stream_new.restype, L.s9_stream_new.argtypes = u64, [vp, cp, ci]
        L.s9_stream_sync.restype, L.s9_stream_sync.argtypes = ci, [vp, u64, dbl, cp, ci]
        L.s7_run_cascade.restype = ci
        L.s7_run_cascade.argtypes = [vp, vp, i64, i64, i64, dbl, vp, vp, ci, vp, vp, dbl, vp, vp, vp, vp, vp, vp,
                                     vp, vp, vp, cp, ci]
        L.s7_tail_launch.restype, L.s7_tail_launch.argtypes = ci, [vp, vp, u64, cp, ci]
        L.s7_run.restype = ci
        L.s7_run.argtypes = [vp, vp, vp, ci, u64, u64, i64, vp, i64, i64, dbl, vp, vp, vp, vp, vp, vp, vp, cp, ci]
        self.L, self.path, self.err = L, path, ctypes.create_string_buffer(4096)

    def _e(self, what):
        return RuntimeError(f"{what}: {self.err.value.decode(errors='replace')}")

    def ctx(self):
        c = self.L.s7_ctx_new(self.err, len(self.err))
        if not c:
            raise self._e("s7_ctx_new")
        return c

    def pm(self, dem_text, m):
        p = self.L.s7_pm_new(dem_text.encode(), m, self.err, len(self.err))
        if not p:
            raise self._e("s7_pm_new")
        return p

    def decode_rows(self, p, det):
        det = np.ascontiguousarray(det, dtype=np.uint8)
        out = np.zeros(len(det), np.uint8)
        self.L.s7_pm_decode_rows(p, det.ctypes.data, len(det), out.ctypes.data)
        return out

    def pers_new(self, ctx, nl):
        dev = ctypes.c_uint64()
        p = self.L.s9_pers_new(ctx, nl, ctypes.byref(dev), self.err, len(self.err))
        if not p:
            raise self._e("s9_pers_new")
        return p, dev.value

    def pers_status(self, p):
        return [int(self.L.s9_pers_status(p, i)) for i in range(16)]

    def pers_stop(self, p, timeout=5.0):
        r = self.L.s9_pers_stop(p, timeout, self.err, len(self.err))
        if r < 0:
            raise self._e("s9_pers_stop")
        return int(r)

    def stream_new(self, ctx):
        s = self.L.s9_stream_new(ctx, self.err, len(self.err))
        if not s:
            raise self._e("s9_stream_new")
        return s

    def stream_sync(self, ctx, s, timeout=10.0):
        """0 when the stream is done; raises on an error; 2 if the resident kernel still runs after timeout s."""
        r = self.L.s9_stream_sync(ctx, s, timeout, self.err, len(self.err))
        if r == 1:
            raise self._e("s9_stream_sync")
        return r

    def tail(self, ctx, stream, logit_ptr):
        """The tail kernel on stream (called inside a CUDA graph capture, after the engine): it publishes
        (launch counter << 32 | the float32 logit's bits) into mapped host memory for s7_run's mode 1."""
        if self.L.s7_tail_launch(ctx, stream, logit_ptr, self.err, len(self.err)):
            raise self._e("s7_tail_launch")

    def run(self, ctx, g, x, pool, det, tau, pm, n, warm=200):
        """One pass of the compiled loop in mode 1 (mapped) on the current stream: warm untimed shots, then shots
        0..n-1 timed. Per shot the bits are copied from pool into the graph's static input x (untimed), then t0, the
        launch of g (the engine and the tail kernel), the spin on the mapped word, the logit, |double(z)| < tau ->
        correlated PM (t1, t2 around the call), else z > 0, t3. -> per shot t_all = t3 - t0 and t_pm = t2 - t1 (ns),
        dec, routed, z."""
        import torch
        out = dict(t_all=np.zeros(n, np.int64), t_pm=np.zeros(n, np.int64), dec=np.zeros(n, np.uint8),
                   routed=np.zeros(n, np.uint8), z=np.zeros(n, np.float32))
        m = det.shape[1]
        assert det.flags.c_contiguous and det.dtype == np.uint8 and pool.is_contiguous() and pool.shape[1] == m
        rc = self.L.s7_run(ctx, g.raw_cuda_graph_exec(), torch.cuda.current_stream().cuda_stream, 1, x.data_ptr(),
                           pool.data_ptr(), m, det.ctypes.data, n, warm, float(tau), pm, None,
                           *(out[k].ctypes.data for k in ("t_all", "t_pm", "dec", "routed", "z")),
                           self.err, len(self.err))
        if rc:
            raise self._e("s7_run mapped")
        return out

    def run_persistent(self, p, det, tau, pm, n, warm=200):
        """One pass of the persistent mode (warm untimed shots, then shots 0..n-1 timed) against the resident kernel;
        a shot with |double(z)| < tau goes to correlated PM (pm; None when tau <= 0), any other is decided by z > 0.
        -> per shot t_all (ns: the request's packing to its decision), t_pm (ns: the correlated-PM call), dec, routed,
        z (the kernel's c); t_dt, z_dt, fb_pm belong to the development fallbacks and stay 0."""
        out = dict(t_all=np.zeros(n, np.int64), t_pm=np.zeros(n, np.int64), t_dt=np.zeros(n, np.int64),
                   dec=np.zeros(n, np.uint8), routed=np.zeros(n, np.uint8), z=np.zeros(n, np.float32),
                   z_dt=np.zeros(n, np.float32), fb_pm=np.zeros(n, np.uint8))
        assert det.flags.c_contiguous and det.dtype == np.uint8 and det.shape[1] == 336 and len(det) >= n
        rc = self.L.s7_run_cascade(p, det.ctypes.data, 336, n, warm, float(tau), pm, None, 0, None, None, 0.0,
                                   *(out[k].ctypes.data for k in ("t_all", "t_pm", "t_dt", "dec", "routed", "z",
                                                                  "z_dt", "fb_pm")),
                                   None, self.err, len(self.err))
        if rc:
            raise self._e(f"s7_run_cascade (rc {rc})")
        return out
