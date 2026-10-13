// The compiled router: the decoder's per-shot routing (the network's logit, then correlated PyMatching when
// |c| < tau) with no Python in the per-shot loop. Built by build.sh into libs7router.so (PyMatching v2.4.0 + Stim
// 1320ad7e linked in statically) and driven through ctypes (engine.py). The CUDA driver API is resolved from
// libcuda.so.1 at run time, so no CUDA headers are needed.
//
// Per shot k in the graph modes (a network launched as a CUDA graph): the shot's bits copied device to device into the
// graph's static input and the stream synchronised (untimed); t0; cuGraphLaunch; completion; the float32 logit z; the
// branch |double(z)| < tau; a routed shot's fired detectors decoded by correlated matching (t1, t2 around the call);
// otherwise the decision z > 0; t3. Completion modes:
//   0 event : the graph ends with the logit's copy into pinned host memory; cuEventRecord after the
//             launch, spin on cuEventQuery, then read the pinned float.
//   1 mapped: the graph ends with s7_tail, which increments a device counter and stores one 64-bit word
//             (counter << 32 | logit bits) into mapped pinned host memory, then fences at system scope. The host spins
//             until the word's counter equals this launch's expected value; value and version come in one aligned
//             8-byte store, and a stale word carries an older counter, so it can never be taken for this launch.
//   persistent (the d = 7 executable): no launch per shot. A resident kernel (kernel.cu) polls a tagged request
//             line in mapped host memory; per shot the host writes the shot's 336 bits into it (t0 before the packing)
//             and spins on the result word (seq << 32 | logit bits) the kernel publishes; then the same branch and
//             correlated PM as above (s7_run_persistent; the s9_* calls own the mapped block and the stop).
// In the studies the mapped mode drove the Triton engine at d = 9-15; this release drives only the persistent mode.
// Built with -DS7_NO_PM (development builds) the library has no matcher: only tau <= 0 passes.
#include <dlfcn.h>
#include <time.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef S7_NO_PM
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "stim/dem/detector_error_model.h"
#endif

namespace {

typedef int CUresult;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUgraphExec;
typedef void *CUgraph;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUcontext;
typedef unsigned long long CUdeviceptr;
const CUresult CUDA_SUCCESS = 0, CUDA_ERROR_NOT_READY = 600;
const unsigned CU_EVENT_DISABLE_TIMING = 0x2, CU_MEMHOSTALLOC_PORTABLE = 0x1, CU_MEMHOSTALLOC_DEVICEMAP = 0x2;

struct Drv {
    CUresult (*cuCtxGetCurrent)(CUcontext *);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuGraphLaunch)(CUgraphExec, CUstream);
    CUresult (*cuEventCreate)(CUevent *, unsigned);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventQuery)(CUevent);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuModuleLoadData)(CUmodule *, const void *);
    CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                               CUstream, void **, void **);
    CUresult (*cuMemHostAlloc)(void **, size_t, unsigned);
    CUresult (*cuMemHostGetDevicePointer)(CUdeviceptr *, void *, unsigned);
    CUresult (*cuMemFreeHost)(void *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemsetD32)(CUdeviceptr, unsigned, size_t);
    CUresult (*cuGetErrorString)(CUresult, const char **);
    CUresult (*cuStreamCreate)(CUstream *, unsigned);
    CUresult (*cuStreamQuery)(CUstream);
    CUresult (*cuGraphInstantiateWithFlags)(CUgraphExec *, CUgraph, unsigned long long);
};

// One thread: the logit (written by the engine's last kernel; this launch has a full dependency on it), the counter,
// one 8-byte store to the mapped host word, a system-scope fence.
const char *TAIL_PTX = R"(
.version 7.0
.target sm_70
.address_size 64
.visible .entry s7_tail(.param .u64 p_logit, .param .u64 p_ctr, .param .u64 p_word)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p_logit];
    ld.param.u64 %rd2, [p_ctr];
    ld.param.u64 %rd3, [p_word];
    cvta.to.global.u64 %rd1, %rd1;
    cvta.to.global.u64 %rd2, %rd2;
    cvta.to.global.u64 %rd3, %rd3;
    ld.volatile.global.b32 %r1, [%rd1];
    ld.volatile.global.u32 %r2, [%rd2];
    add.u32 %r3, %r2, 1;
    st.volatile.global.u32 [%rd2], %r3;
    cvt.u64.u32 %rd4, %r3;
    shl.b64 %rd5, %rd4, 32;
    cvt.u64.u32 %rd6, %r1;
    or.b64 %rd7, %rd5, %rd6;
    st.volatile.global.u64 [%rd3], %rd7;
    membar.sys;
    ret;
}
)";

struct Ctx {
    Drv d{};
    void *lib = nullptr;
    CUevent ev = nullptr;
    CUmodule mod = nullptr;
    CUfunction tail = nullptr;
    CUdeviceptr ctr = 0;          // the tail kernel's launch counter (device memory)
    uint64_t *word = nullptr;     // mapped pinned host word (counter << 32 | logit bits)
    CUdeviceptr word_dev = 0;
    int flush = 0;                // WDDM (Windows, WSL2) batches launches; cuStreamQuery after each launch submits it
    std::string err;
};

#ifndef S7_NO_PM
struct PM {
    std::unique_ptr<pm::UserGraph> g;
    pm::Mwpm *mwpm = nullptr;
    size_t m = 0;
    std::vector<uint64_t> ev;
    std::vector<uint8_t> obs;
};
#else
struct PM {};
#endif

inline int64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (int64_t)t.tv_sec * 1000000000LL + t.tv_nsec;
}

// The wheel's decode(z, enable_correlations=True): fired detectors in ascending order, then
// pm::decode_detection_events on get_mwpm_with_search_graph() with edge correlations.
#ifndef S7_NO_PM
inline int pm_decode_row(PM *p, const uint8_t *row) {
    p->ev.clear();
    for (size_t i = 0; i < p->m; i++)
        if (row[i]) p->ev.push_back(i);
    std::fill(p->obs.begin(), p->obs.end(), 0);
    pm::total_weight_int w = 0;
    pm::decode_detection_events(*p->mwpm, p->ev, p->obs.data(), w, true);
    return p->obs[0];
}
#else
inline int pm_decode_row(PM *, const uint8_t *) { return 0; }
#endif

bool ck(Ctx *c, CUresult r, const char *what) {
    if (r == CUDA_SUCCESS) return true;
    const char *s = nullptr;
    if (c->d.cuGetErrorString) c->d.cuGetErrorString(r, &s);
    c->err = std::string(what) + " failed: " + std::to_string(r) + " " + (s ? s : "");
    return false;
}

void put_err(const std::string &e, char *err, int errlen) {
    if (err && errlen > 0) snprintf(err, (size_t)errlen, "%s", e.c_str());
}

// ------------------------------------------------------------------------------------------------ the persistent mode
// One 4 KB mapped pinned block (allocated and first touched by the calling, pinned thread): NL request lines of 16
// words at 0, the result word at 1024 (its own line), the status line at 1088, z / c of the last shot at 1152 (checks).
const uint32_t S9_CMD_STOP = 1, S9_READY = 1, S9_STOPPED = 2, S9_WATCHDOG = 3;
struct Pers {
    Ctx *c = nullptr;
    void *block = nullptr;
    uint32_t *req = nullptr;
    uint64_t *res = nullptr;
    uint32_t *status = nullptr;
    float *zout = nullptr;
    CUdeviceptr dev = 0;
    int nl = 1;
    uint32_t seq = 0, tag = 0;    // the last request written
};

inline uint32_t next_seq(uint32_t s) { return s + 1u == 0u ? 1u : s + 1u; }
inline uint32_t next_tag(uint32_t t) { return t >= 255u ? 1u : t + 1u; }

// the 336 detector bytes (0 / 1) -> 14 words of 24 bits (word i bit b = detector 24 i + b): 8 bytes per multiply
inline void s9_pack(const uint8_t *row, uint32_t *w) {
    uint32_t g[42];
    for (int i = 0; i < 42; i++) {
        uint64_t x;
        std::memcpy(&x, row + 8 * i, 8);
        g[i] = (uint32_t)((x * 0x0102040810204080ULL) >> 56);
    }
    for (int i = 0; i < 14; i++) w[i] = g[3 * i] | g[3 * i + 1] << 8 | g[3 * i + 2] << 16;
}

// every copy of the request line: 14 payload words and the command word, each one aligned 32-bit atomic store
// carrying the tag; the command word last with release (the classic mode's doorbell)
inline void s9_send(Pers *P, const uint32_t *w, uint32_t cmd) {
    const uint32_t t = P->tag << 24;
    for (int l = 0; l < P->nl; l++) {
        uint32_t *line = P->req + 16 * l;
        for (int i = 0; i < 14; i++) __atomic_store_n(line + i, t | w[i], __ATOMIC_RELAXED);
        __atomic_store_n(line + 14, t | cmd, __ATOMIC_RELEASE);
    }
}

inline void s9_pause(int64_t ns) {
    if (ns <= 0) return;
    const int64_t t = now_ns();
    if (ns > 2000000) {
        timespec q{(time_t)((ns - 1000000) / 1000000000LL), (long)((ns - 1000000) % 1000000000LL)};
        nanosleep(&q, nullptr);
    }
    while (now_ns() - t < ns) {
    }
}

}  // namespace

#pragma GCC visibility push(default)   // the s7_* C API is the library's only export (build.sh hides the rest)
extern "C" {

// --------------------------------------------------------------------------------------------------- correlated PM
#ifndef S7_NO_PM
void *s7_pm_new(const char *dem_text, int64_t m, char *err, int errlen) {
    try {
        stim::DetectorErrorModel dem(dem_text);
        auto p = new PM();
        p->g = std::make_unique<pm::UserGraph>(
            pm::detector_error_model_to_user_graph(dem, /*enable_correlations=*/true, pm::NUM_DISTINCT_WEIGHTS));
        p->mwpm = &p->g->get_mwpm_with_search_graph();
        p->m = (size_t)m;
        p->obs.assign(std::max<size_t>(1, p->g->get_num_observables()), 0);
        p->ev.reserve((size_t)m);
        if (p->g->get_num_detectors() > (size_t)m) {
            put_err("the DEM has more detectors than the row length", err, errlen);
            delete p;
            return nullptr;
        }
        return p;
    } catch (const std::exception &e) {
        put_err(e.what(), err, errlen);
        return nullptr;
    }
}

void s7_pm_free(void *p) { delete (PM *)p; }

int64_t s7_pm_num_detectors(void *p) { return (int64_t)((PM *)p)->g->get_num_detectors(); }

// decisions of n dense rows (n, m), for the equivalence check against the wheel
int s7_pm_decode_rows(void *p, const uint8_t *det, int64_t n, uint8_t *out) {
    PM *q = (PM *)p;
    for (int64_t i = 0; i < n; i++) out[i] = (uint8_t)pm_decode_row(q, det + i * (int64_t)q->m);
    return 0;
}

// the official single-shot protocol with the direct call: warm-up decodes, then one timed decode per row (ns)
int s7_pm_time(void *p, const uint8_t *det, int64_t n, int64_t warm, int64_t *ts) {
    PM *q = (PM *)p;
    for (int64_t i = 0; i < std::min(warm, n); i++) pm_decode_row(q, det + i * (int64_t)q->m);
    for (int64_t i = 0; i < n; i++) {
        int64_t t0 = now_ns();
        pm_decode_row(q, det + i * (int64_t)q->m);
        ts[i] = now_ns() - t0;
    }
    return 0;
}
#endif

// ---------------------------------------------------------------------------------------------------- the driver
void *s7_ctx_new(char *err, int errlen) {
    auto c = new Ctx();
    c->lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!c->lib) {
        put_err(std::string("dlopen libcuda.so.1: ") + dlerror(), err, errlen);
        delete c;
        return nullptr;
    }
    struct {
        void **slot;
        const char *name;
    } syms[] = {
        {(void **)&c->d.cuCtxGetCurrent, "cuCtxGetCurrent"},
        {(void **)&c->d.cuCtxSynchronize, "cuCtxSynchronize"},
        {(void **)&c->d.cuGraphLaunch, "cuGraphLaunch"},
        {(void **)&c->d.cuEventCreate, "cuEventCreate"},
        {(void **)&c->d.cuEventDestroy, "cuEventDestroy_v2"},
        {(void **)&c->d.cuEventRecord, "cuEventRecord"},
        {(void **)&c->d.cuEventQuery, "cuEventQuery"},
        {(void **)&c->d.cuStreamSynchronize, "cuStreamSynchronize"},
        {(void **)&c->d.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2"},
        {(void **)&c->d.cuModuleLoadData, "cuModuleLoadData"},
        {(void **)&c->d.cuModuleGetFunction, "cuModuleGetFunction"},
        {(void **)&c->d.cuLaunchKernel, "cuLaunchKernel"},
        {(void **)&c->d.cuMemHostAlloc, "cuMemHostAlloc"},
        {(void **)&c->d.cuMemHostGetDevicePointer, "cuMemHostGetDevicePointer_v2"},
        {(void **)&c->d.cuMemFreeHost, "cuMemFreeHost"},
        {(void **)&c->d.cuMemAlloc, "cuMemAlloc_v2"},
        {(void **)&c->d.cuMemFree, "cuMemFree_v2"},
        {(void **)&c->d.cuMemsetD32, "cuMemsetD32_v2"},
        {(void **)&c->d.cuGetErrorString, "cuGetErrorString"},
        {(void **)&c->d.cuStreamCreate, "cuStreamCreate"},
        {(void **)&c->d.cuStreamQuery, "cuStreamQuery"},
        {(void **)&c->d.cuGraphInstantiateWithFlags, "cuGraphInstantiateWithFlags"},
    };
    for (auto &s : syms) {
        *s.slot = dlsym(c->lib, s.name);
        if (!*s.slot) {
            put_err(std::string("libcuda.so.1 has no ") + s.name, err, errlen);
            delete c;
            return nullptr;
        }
    }
    CUcontext cur = nullptr;
    // torch has made its primary context current on this thread; every call below must run on this thread
    if (!ck(c, c->d.cuCtxGetCurrent(&cur), "cuCtxGetCurrent") || !cur) {
        put_err(c->err.empty() ? "no current CUDA context (initialise torch first)" : c->err, err, errlen);
        delete c;
        return nullptr;
    }
    void *w = nullptr;
    bool ok = ck(c, c->d.cuEventCreate(&c->ev, CU_EVENT_DISABLE_TIMING), "cuEventCreate") &&
              ck(c, c->d.cuModuleLoadData(&c->mod, TAIL_PTX), "cuModuleLoadData") &&
              ck(c, c->d.cuModuleGetFunction(&c->tail, c->mod, "s7_tail"), "cuModuleGetFunction") &&
              ck(c, c->d.cuMemAlloc(&c->ctr, 4), "cuMemAlloc") && ck(c, c->d.cuMemsetD32(c->ctr, 0, 1), "cuMemsetD32") &&
              ck(c, c->d.cuMemHostAlloc(&w, 8, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP), "cuMemHostAlloc");
    if (ok) {
        c->word = (uint64_t *)w;
        __atomic_store_n(c->word, (uint64_t)0, __ATOMIC_SEQ_CST);
        ok = ck(c, c->d.cuMemHostGetDevicePointer(&c->word_dev, w, 0), "cuMemHostGetDevicePointer") &&
             ck(c, c->d.cuCtxSynchronize(), "cuCtxSynchronize");
    }
    if (!ok) {
        put_err(c->err, err, errlen);
        return nullptr;  // leaked on purpose: the process stops on this error
    }
    return c;
}

// On WDDM (Windows, WSL2) the driver batches launches and submits them at a later flush; with flush set, the
// mapped-mode loops call cuStreamQuery right after each launch (inside the timed span), as an event query would
void s7_set_flush(void *ctx, int on) { ((Ctx *)ctx)->flush = on ? 1 : 0; }

// torch before 2.9 exposes no cudaGraphExec_t (CUDAGraph.raw_cuda_graph_exec); the router instantiates its own
// executable of torch's kept graph (CUDAGraph(keep_graph=True).raw_cuda_graph()), which stays valid while torch's
// graph object (and so its memory pool) lives
int s7_graph_instantiate(void *ctx, uint64_t graph, uint64_t *exec_out, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    CUgraphExec e = nullptr;
    if (!ck(c, c->d.cuGraphInstantiateWithFlags(&e, (CUgraph)graph, 0), "cuGraphInstantiateWithFlags")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    *exec_out = (uint64_t)e;
    return 0;
}

// the tail kernel on `stream` (called inside torch's graph capture, after the engine)
int s7_tail_launch(void *ctx, void *stream, uint64_t logit_dev, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    CUdeviceptr a = logit_dev, b = c->ctr, w = c->word_dev;
    void *args[] = {&a, &b, &w};
    if (!ck(c, c->d.cuLaunchKernel(c->tail, 1, 1, 1, 1, 1, 1, 0, (CUstream)stream, args, nullptr), "cuLaunchKernel")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    return 0;
}

uint64_t s7_word(void *ctx) { return __atomic_load_n(((Ctx *)ctx)->word, __ATOMIC_ACQUIRE); }

// One pass: `warm` untimed shots (rows 0 .. warm-1), then all n shots timed. mode 0 event (h_logit = the pinned float
// the graph copies into), 1 mapped. pm may be null when tau <= 0 (nothing can be routed). Outputs per shot: t_all,
// t_pm (ns; 0 when not routed), dec, routed, z.
int s7_run(void *ctx, void *exec, void *stream, int mode, uint64_t x_dev, uint64_t pool_dev, int64_t m,
           const uint8_t *det, int64_t n, int64_t warm, double tau, void *pmp, const float *h_logit, int64_t *t_all,
           int64_t *t_pm, uint8_t *dec, uint8_t *routed, float *z, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    PM *p = (PM *)pmp;
    const Drv &d = c->d;
    CUstream st = (CUstream)stream;
    if (tau > 0 && !p) {
        put_err("tau > 0 needs a matcher", err, errlen);
        return 1;
    }
    if (!ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    uint32_t seq = (uint32_t)(s7_word(ctx) >> 32);  // launches of the mapped graph so far (all synchronised)
    for (int64_t it = -warm; it < n; it++) {
        const int64_t k = it < 0 ? (it + warm) % n : it;
        if (!ck(c, d.cuMemcpyDtoDAsync((CUdeviceptr)x_dev, (CUdeviceptr)(pool_dev + (uint64_t)(k * m)), (size_t)m, st),
                "cuMemcpyDtoDAsync") ||
            !ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
            put_err(c->err, err, errlen);
            return 1;
        }
        const uint32_t expect = seq + 1;
        float zk;
        const int64_t t0 = now_ns();
        CUresult r = d.cuGraphLaunch((CUgraphExec)exec, st);
        if (r != CUDA_SUCCESS) {
            ck(c, r, "cuGraphLaunch");
            put_err(c->err, err, errlen);
            return 1;
        }
        if (c->flush && mode == 1) d.cuStreamQuery(st);   // WDDM: submit the batched launch now
        if (mode == 0) {
            r = d.cuEventRecord(c->ev, st);
            if (r == CUDA_SUCCESS)
                while ((r = d.cuEventQuery(c->ev)) == CUDA_ERROR_NOT_READY) {
                }
            if (r != CUDA_SUCCESS) {
                ck(c, r, "cuEventRecord / cuEventQuery");
                put_err(c->err, err, errlen);
                return 1;
            }
            zk = *(const volatile float *)h_logit;
        } else {
            uint64_t w;
            uint64_t spins = 0;
            while ((uint32_t)((w = __atomic_load_n(c->word, __ATOMIC_ACQUIRE)) >> 32) != expect) {
                if ((++spins & 0xffff) == 0 && now_ns() - t0 > 2000000000LL) {   // 2 s: the tail never arrived
                    put_err("mapped word not updated within 2 s (counter " + std::to_string((uint32_t)(w >> 32)) +
                                ", expected " + std::to_string(expect) + ")", err, errlen);
                    return 1;
                }
            }
            const uint32_t bits = (uint32_t)w;
            std::memcpy(&zk, &bits, 4);
            seq = expect;
        }
        const double zd = (double)zk;
        const bool rt = std::fabs(zd) < tau;
        int dk;
        int64_t t1 = 0, t2 = 0;
        if (rt) {
            t1 = now_ns();
            dk = pm_decode_row(p, det + k * m);
            t2 = now_ns();
        } else {
            dk = zd > 0;
        }
        const int64_t t3 = now_ns();
        if (it >= 0) {
            t_all[k] = t3 - t0;
            t_pm[k] = t2 - t1;
            dec[k] = (uint8_t)dk;
            routed[k] = (uint8_t)rt;
            z[k] = zk;
        }
    }
    if (!ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    return 0;
}

// Development option, not used by the studies' decoder or this release: s7_run's mapped mode with a second-tier
// decoder ("DT", a small network for the routed shots) behind a launched network (GPUs without clusters): the
// network's graph and the DT's graph are captured on the same static input x, each with the tail word, so a routed shot
// needs no second copy; fb 1 (dt) decides by z_DT > 0, fb 2 (dt2) sends |z_DT| < tau2 on to correlated PM. As s7_run,
// the shot's copy into x is outside the timed span; t_dt is the DT graph's launch to its word.
static bool wait_word(Ctx *c, uint32_t expect, int64_t t0, uint64_t *w, char *err, int errlen) {
    uint64_t spins = 0;
    while ((uint32_t)((*w = __atomic_load_n(c->word, __ATOMIC_ACQUIRE)) >> 32) != expect) {
        if ((++spins & 0xffff) == 0 && now_ns() - t0 > 2000000000LL) {   // 2 s: the tail never arrived
            put_err("mapped word not updated within 2 s (counter " + std::to_string((uint32_t)(*w >> 32)) +
                        ", expected " + std::to_string(expect) + ")", err, errlen);
            return false;
        }
    }
    return true;
}

int s7_run_dt(void *ctx, void *exec, void *exec_dt, void *stream, uint64_t x_dev, uint64_t pool_dev, int64_t m,
              const uint8_t *det, int64_t n, int64_t warm, double tau, double tau2, int fb, void *pmp, int64_t *t_all,
              int64_t *t_pm, int64_t *t_dt, uint8_t *dec, uint8_t *routed, float *z, float *z_dt, uint8_t *fb_pm,
              char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    PM *p = (PM *)pmp;
    const Drv &d = c->d;
    CUstream st = (CUstream)stream;
    if ((fb != 1 && fb != 2) || !exec_dt || (fb == 2 && !p)) {
        put_err("s7_run_dt: fb 1 (dt) or 2 (dt2, with a matcher) and the DT's graph", err, errlen);
        return 1;
    }
    if (!ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    uint32_t seq = (uint32_t)(s7_word(ctx) >> 32);  // the tail word's launches so far (all synchronised)
    const float nan = std::nanf("");
    for (int64_t it = -warm; it < n; it++) {
        const int64_t k = it < 0 ? (it + warm) % n : it;
        if (!ck(c, d.cuMemcpyDtoDAsync((CUdeviceptr)x_dev, (CUdeviceptr)(pool_dev + (uint64_t)(k * m)), (size_t)m, st),
                "cuMemcpyDtoDAsync") ||
            !ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
            put_err(c->err, err, errlen);
            return 1;
        }
        const int64_t t0 = now_ns();
        CUresult r = d.cuGraphLaunch((CUgraphExec)exec, st);
        if (r != CUDA_SUCCESS) {
            ck(c, r, "cuGraphLaunch");
            put_err(c->err, err, errlen);
            return 1;
        }
        if (c->flush) d.cuStreamQuery(st);                  // WDDM: submit the batched launch now
        uint64_t w;
        if (!wait_word(c, seq + 1, t0, &w, err, errlen)) return 1;
        seq += 1;
        float zk;
        const uint32_t bits = (uint32_t)w;
        std::memcpy(&zk, &bits, 4);
        const double zd = (double)zk;
        const bool rt = std::fabs(zd) < tau;
        int dk;
        int64_t t1 = 0, t2 = 0, ta = 0, tb = 0;
        float zdt = nan;
        bool fp = false;
        if (rt) {
            ta = now_ns();
            r = d.cuGraphLaunch((CUgraphExec)exec_dt, st);
            if (r != CUDA_SUCCESS) {
                ck(c, r, "cuGraphLaunch (DT)");
                put_err(c->err, err, errlen);
                return 1;
            }
            if (c->flush) d.cuStreamQuery(st);
            if (!wait_word(c, seq + 1, ta, &w, err, errlen)) return 1;
            seq += 1;
            const uint32_t lb = (uint32_t)w;
            std::memcpy(&zdt, &lb, 4);
            tb = now_ns();
            if (fb == 2 && std::fabs((double)zdt) < tau2) {
                fp = true;
                t1 = now_ns();
                dk = pm_decode_row(p, det + k * m);
                t2 = now_ns();
            } else {
                dk = (double)zdt > 0;
            }
        } else {
            dk = zd > 0;
        }
        const int64_t t3 = now_ns();
        if (it >= 0) {
            t_all[k] = t3 - t0;
            t_pm[k] = t2 - t1;
            t_dt[k] = tb - ta;
            dec[k] = (uint8_t)dk;
            routed[k] = (uint8_t)rt;
            z[k] = zk;
            z_dt[k] = zdt;
            fb_pm[k] = (uint8_t)fp;
        }
    }
    if (!ck(c, d.cuStreamSynchronize(st), "cuStreamSynchronize")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    return 0;
}


// ------------------------------------------------------------------------------------------------ the persistent mode
// the mapped block -> its device address (request lines at +0, result +1024, status +1088, z / c +1152)
void *s9_pers_new(void *ctx, int nl, uint64_t *dev, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    if (nl < 1 || nl > 8) {
        put_err("nl must be 1..8", err, errlen);
        return nullptr;
    }
    auto P = new Pers();
    P->c = c;
    P->nl = nl;
    if (!ck(c, c->d.cuMemHostAlloc(&P->block, 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
            "cuMemHostAlloc") ||
        !ck(c, c->d.cuMemHostGetDevicePointer(&P->dev, P->block, 0), "cuMemHostGetDevicePointer")) {
        put_err(c->err, err, errlen);
        delete P;
        return nullptr;
    }
    std::memset(P->block, 0, 4096);                  // first touch on this thread's NUMA node
    char *b = (char *)P->block;
    P->req = (uint32_t *)b;
    P->res = (uint64_t *)(b + 1024);
    P->status = (uint32_t *)(b + 1088);
    P->zout = (float *)(b + 1152);
    *dev = (uint64_t)P->dev;
    return P;
}

void s9_pers_free(void *pp) {
    Pers *P = (Pers *)pp;
    if (P->block) P->c->d.cuMemFreeHost(P->block);
    delete P;
}

// before a launch: every request line empty (tag 0), result and status zero; seq0 / tag0 the launch's parameters
void s9_pers_reset(void *pp, uint32_t seq0, uint32_t tag0) {
    Pers *P = (Pers *)pp;
    for (int i = 0; i < 16 * 8; i++) __atomic_store_n(P->req + i, 0u, __ATOMIC_RELAXED);
    __atomic_store_n(P->res, (uint64_t)0, __ATOMIC_RELAXED);
    for (int i = 0; i < 16; i++) __atomic_store_n(P->status + i, 0u, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    P->seq = seq0;
    P->tag = tag0;
}

uint32_t s9_pers_status(void *pp, int i) { return __atomic_load_n(((Pers *)pp)->status + i, __ATOMIC_ACQUIRE); }
uint64_t s9_pers_result(void *pp) { return __atomic_load_n(((Pers *)pp)->res, __ATOMIC_ACQUIRE); }
uint32_t s9_pers_seq(void *pp) { return ((Pers *)pp)->seq; }
uint32_t s9_pers_tag(void *pp) { return ((Pers *)pp)->tag; }

// spin until status[0] is one of want (bit mask of states), at most timeout_s; the elapsed ns, or -1 on the timeout
int64_t s9_pers_wait(void *pp, uint32_t want_mask, double timeout_s) {
    Pers *P = (Pers *)pp;
    const int64_t t0 = now_ns(), lim = (int64_t)(timeout_s * 1e9);
    for (;;) {
        const uint32_t st = __atomic_load_n(P->status, __ATOMIC_ACQUIRE);
        if (st < 32 && ((want_mask >> st) & 1u)) return now_ns() - t0;
        if (now_ns() - t0 > lim) return -1;
    }
}

// the stop request; then the kernel's final state (STOPPED or WATCHDOG) within timeout_s -> the requests it served,
// or -1 (state never came; err says which)
int64_t s9_pers_stop(void *pp, double timeout_s, char *err, int errlen) {
    Pers *P = (Pers *)pp;
    P->seq = next_seq(P->seq);
    P->tag = next_tag(P->tag);
    uint32_t w[14] = {0};
    s9_send(P, w, S9_CMD_STOP);
    if (s9_pers_wait(pp, (1u << S9_STOPPED) | (1u << S9_WATCHDOG), timeout_s) < 0) {
        put_err("no stop / watchdog state within " + std::to_string(timeout_s) + " s (state " +
                    std::to_string(s9_pers_status(pp, 0)) + ")", err, errlen);
        return -1;
    }
    return (int64_t)s9_pers_status(pp, 1);
}

// a non-blocking stream for the resident kernel (it never synchronises with the legacy default stream)
uint64_t s9_stream_new(void *ctx, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    CUstream s = nullptr;
    if (!ck(c, c->d.cuStreamCreate(&s, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate")) {
        put_err(c->err, err, errlen);
        return 0;
    }
    return (uint64_t)s;
}

// the resident kernel's stream done within timeout_s (cuStreamQuery, so a kernel that never ends cannot hang the host):
// 0 done, 1 an error, 2 still running at the timeout
int s9_stream_sync(void *ctx, uint64_t stream, double timeout_s, char *err, int errlen) {
    Ctx *c = (Ctx *)ctx;
    const int64_t t0 = now_ns(), lim = (int64_t)(timeout_s * 1e9);
    for (;;) {
        const CUresult r = c->d.cuStreamQuery((CUstream)stream);
        if (r == CUDA_SUCCESS) return 0;
        if (r != CUDA_ERROR_NOT_READY) {
            ck(c, r, "cuStreamQuery");
            put_err(c->err, err, errlen);
            return 1;
        }
        if (now_ns() - t0 > lim) {
            put_err("the resident kernel has not ended after " + std::to_string(timeout_s) + " s", err, errlen);
            return 2;
        }
    }
}

// The persistent completion mode: `warm` untimed shots (rows 0 .. warm-1), then all n shots timed.
// Per shot: an optional untimed pause (pause_ns, the delay test), next seq / tag, t0, the packing and the request
// stores, the spin on the result word (2 s timeout), z, the branch |double(z)| < tau; a routed shot goes to the
// fallback fb: 0 correlated PM (t1, t2 around the call; the studies' setting); 1 the second-tier decoder DT (a
// development option, see s7_run_dt): its graph (exec_dt, captured with s7_tail's word, its static input the buffer
// the kernel staged the shot's bits into) launched on stream_dt, a spin on the mapped word (2 s), the decision
// z_DT > 0; 2 the same, then correlated PM when |double(z_DT)| < tau2. Unrouted: z > 0;
// t3. t_pm is the correlated-PM call only (the Grace-scaled share), t_dt the DT's launch to its word. zrow (n x 8,
// optional): the kernel's z_0..z_6 and c (checks; written before the result). z_dt (NaN when not routed), fb_pm (sent
// on to PM by the DT tier).
int s7_run_cascade(void *pp, const uint8_t *det, int64_t m, int64_t n, int64_t warm, double tau, void *pmp,
                   const int64_t *pause_ns, int fb, void *exec_dt, void *stream_dt, double tau2, int64_t *t_all,
                   int64_t *t_pm, int64_t *t_dt, uint8_t *dec, uint8_t *routed, float *z, float *z_dt, uint8_t *fb_pm,
                   float *zrow, char *err, int errlen) {
    Pers *P = (Pers *)pp;
    Ctx *c = P->c;
    PM *p = (PM *)pmp;
    if (m != 336) {
        put_err("the persistent mode is d = 7 (336 detectors)", err, errlen);
        return 1;
    }
    if (tau > 0 && fb != 1 && !p) {
        put_err("tau > 0 with a correlated-PM tier needs a matcher", err, errlen);
        return 1;
    }
    if (fb >= 1 && tau > 0 && !exec_dt) {
        put_err("a DT fallback needs the DT graph", err, errlen);
        return 1;
    }
    if (s9_pers_status(pp, 0) != S9_READY) {
        put_err("the resident kernel is not ready (state " + std::to_string(s9_pers_status(pp, 0)) + ")", err, errlen);
        return 1;
    }
    if (fb >= 1 && stream_dt && !ck(c, c->d.cuStreamSynchronize((CUstream)stream_dt), "cuStreamSynchronize")) {
        put_err(c->err, err, errlen);
        return 1;
    }
    uint32_t dseq = (uint32_t)(__atomic_load_n(c->word, __ATOMIC_ACQUIRE) >> 32);   // the DT graph's launches so far
    const float nan = std::nanf("");
    for (int64_t it = -warm; it < n; it++) {
        const int64_t k = it < 0 ? (it + warm) % n : it;
        if (pause_ns && it >= 0) s9_pause(pause_ns[k]);
        P->seq = next_seq(P->seq);
        P->tag = next_tag(P->tag);
        const int64_t t0 = now_ns();
        uint32_t w[14];
        s9_pack(det + k * m, w);
        s9_send(P, w, 0);
        uint64_t wd;
        uint64_t spins = 0;
        while ((uint32_t)((wd = __atomic_load_n(P->res, __ATOMIC_ACQUIRE)) >> 32) != P->seq) {
            if ((++spins & 0xffff) == 0 && now_ns() - t0 > 2000000000LL) {   // 2 s: no result
                put_err("no result within 2 s (seq " + std::to_string(P->seq) + ", word seq " +
                            std::to_string((uint32_t)(wd >> 32)) + ", state " + std::to_string(s9_pers_status(pp, 0)) +
                            ")", err, errlen);
                return 2;
            }
        }
        const uint32_t bits = (uint32_t)wd;
        float zk;
        std::memcpy(&zk, &bits, 4);
        const double zd = (double)zk;
        const bool rt = std::fabs(zd) < tau;
        int dk;
        int64_t t1 = 0, t2 = 0, ta = 0, tb = 0;
        float zdt = nan;
        bool fp = false;
        if (rt && fb == 0) {
            t1 = now_ns();
            dk = pm_decode_row(p, det + k * m);
            t2 = now_ns();
        } else if (rt) {
            ta = now_ns();
            CUresult r = c->d.cuGraphLaunch((CUgraphExec)exec_dt, (CUstream)stream_dt);
            if (r != CUDA_SUCCESS) {
                ck(c, r, "cuGraphLaunch (DT)");
                put_err(c->err, err, errlen);
                return 1;
            }
            const uint32_t expect = dseq + 1;
            uint64_t wv;
            uint64_t sp = 0;
            while ((uint32_t)((wv = __atomic_load_n(c->word, __ATOMIC_ACQUIRE)) >> 32) != expect) {
                if ((++sp & 0xffff) == 0 && now_ns() - ta > 2000000000LL) {
                    put_err("the DT's mapped word not updated within 2 s (counter " + std::to_string((uint32_t)(wv >> 32)) +
                                ", expected " + std::to_string(expect) + ")", err, errlen);
                    return 2;
                }
            }
            dseq = expect;
            const uint32_t lb = (uint32_t)wv;
            std::memcpy(&zdt, &lb, 4);
            tb = now_ns();
            if (fb == 2 && std::fabs((double)zdt) < tau2) {
                fp = true;
                t1 = now_ns();
                dk = pm_decode_row(p, det + k * m);
                t2 = now_ns();
            } else {
                dk = (double)zdt > 0;
            }
        } else {
            dk = zd > 0;
        }
        const int64_t t3 = now_ns();
        if (it >= 0) {
            t_all[k] = t3 - t0;
            t_pm[k] = t2 - t1;
            if (t_dt) t_dt[k] = tb - ta;
            dec[k] = (uint8_t)dk;
            routed[k] = (uint8_t)rt;
            z[k] = zk;
            if (z_dt) z_dt[k] = zdt;
            if (fb_pm) fb_pm[k] = (uint8_t)fp;
            if (zrow) std::memcpy(zrow + 8 * k, P->zout, 32);
        }
    }
    return 0;
}

// the entry point with correlated PM as the only fallback (the studies' runs)
int s7_run_persistent(void *pp, const uint8_t *det, int64_t m, int64_t n, int64_t warm, double tau, void *pmp,
                      const int64_t *pause_ns, int64_t *t_all, int64_t *t_pm, uint8_t *dec, uint8_t *routed, float *z,
                      float *zrow, char *err, int errlen) {
    return s7_run_cascade(pp, det, m, n, warm, tau, pmp, pause_ns, 0, nullptr, nullptr, 0.0, t_all, t_pm, nullptr, dec,
                          routed, z, nullptr, nullptr, zrow, err, errlen);
}

// one request that expects no answer within timeout_s (the timeout test): 0 if none came, 1 if one did
int s9_probe(void *pp, const uint8_t *row, double timeout_s) {
    Pers *P = (Pers *)pp;
    P->seq = next_seq(P->seq);
    P->tag = next_tag(P->tag);
    uint32_t w[14];
    s9_pack(row, w);
    s9_send(P, w, 0);
    const int64_t t0 = now_ns(), lim = (int64_t)(timeout_s * 1e9);
    while (now_ns() - t0 < lim)
        if ((uint32_t)(__atomic_load_n(P->res, __ATOMIC_ACQUIRE) >> 32) == P->seq) return 1;
    return 0;
}

}  // extern "C"
#pragma GCC visibility pop
