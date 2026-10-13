// The d = 7 ConvLines network (one block, width 64) as a persistent, weight-resident CUDA kernel for sm_90a (H100,
// H200): the executable whose decisions and GPU time the studies registered at d = 7. One 7-CTA thread-block cluster
// stays resident; each CTA copies its tables and every weight into shared memory once, then serves requests with 12
// compute warps plus one service warp:
//   service warp   polls the request line in mapped host memory (lanes 0-14 one 32-bit word each, ld.relaxed.sys)
//                  until every word carries the request's tag; unpacks the 336 detector bits into the CTA's shared
//                  bits; releases the compute warps (named barrier GO); computes the CTA's line parities into CTA 0
//                  (off the critical path) and joins the body's cluster barriers
//   compute warps  the network: stem, block(s), head, pooled partials, read-out of line `rank`, z gathered on CTA 0;
//                  a named barrier of the 384 compute threads stands in for __syncthreads. CTA 0's thread 0 forms c
//                  and publishes (seq << 32 | c bits) with one st.release.sys.u64 into mapped host memory.
// The compute warps run the instructions of an earlier one-launch-per-shot kernel of the same network in the same order
// ("the launch kernel" below; not included); the studies checked this kernel's c against it bit for bit.
// Request line (16 words, 64 bytes): words 0-13 bits 0-23 the detectors 24 i .. 24 i + 23, word 14 bits 0-23 the
// command (0 decode, 1 stop), bits 24-31 of every word the request's tag (1..255; 0 = empty). The host writes NL copies
// (64-byte stride) so that a service warp can keep up to 4 staggered polls in flight (poll i reads copy i % NL).
// p.fwd = 1 (the deployed setting): only CTA 0's service warp polls host memory; it forwards the accepted words (tags
// kept) into the other CTAs' shared slot `fwd` (DSMEM stores), where their service warps poll locally (in development,
// seven CTAs polling one host line slowed each other down). A watchdog in CTA 0 is forwarded as command 2.
// BODY 1 is the engine; BODY 0 (detect, one cluster barrier, publish 0) and BODY 2 (CTA 0 publishes on its own
// detection) are handshake microbenchmarks. With p.check the empty bodies publish the received bits' checksum
// sum_i (i + 1) bit_i instead of 0 (protocol tests). -DS9_NOCLUSTER (sm_80) builds only the empty bodies, without
// clusters or bulk copies (CTA 0 serves alone; the other six leave after the prologue): the host protocol's tests on
// GPUs without clusters.
//   nvcc -O3 -std=c++17 -arch=sm_90a -cubin -o kernel_sm90a.cubin kernel.cu        (timing/d7/build.sh)
#include <cuda_fp16.h>
#include <stdint.h>

#include "dtc.cuh"

namespace s9 {
using namespace dtc;
constexpr int D = 7, M = 336, NX = 192, NC = 7, RC = 48, RP = 48, MT = 3, NTH = 384;   // the compute warps
constexpr int WS = NTH / 32, NT = NTH + 32;                                            // the service warp; threads
constexpr int C = 64, CB = 16, FF = 128, KSP = 80, KB = 27, NB = 32, NS = 32;  // NS: score columns h 8 + j
constexpr int ZB = 336, ZR = 336;                     // a zero byte of the bits, the zero row of u
constexpr int SA0 = 88, SWB = 88, SWD = 72, SWC = 24, SWU = 24, SWS = 72, SWV = 72, SY = 72, SV = 24, SG = 40;  // strides
constexpr int T0B = RP * KSP * 2 + 128 + RP * C * 4, T1B = RP * NB * 2 + RP * NS * 4 + RP * 8 * 4;   // table bytes
constexpr int UB = (M + 1) * CB * 2;                  // one u buffer (bytes)
constexpr int NW = 15, LINE = 16;                     // words used / words per request line
constexpr uint32_t CMD_STOP = 1, CMD_WATCHDOG = 2;
constexpr uint32_t ST_READY = 1, ST_STOPPED = 2, ST_WATCHDOG = 3;
constexpr int BAR_ALL = 4, BAR_GO = 5;                // named barriers (1-3: the tiles' groups)
constexpr int NDBG = 64;                              // stamp ring (shots)

struct Params {
  const uint32_t* req;   // (NL, 16) mapped request lines
  unsigned long long* res;   // mapped result word (seq << 32 | c bits)
  uint32_t* status;      // mapped status line: [0] state, [1] requests served, [2] last seq, [3] last tag,
                         // [4] classic-mode payload words that lagged the doorbell
  float* zout;           // optional (mapped host, 8): the last shot's z_0..z_6 and c, written before the result
  const char* T0;        // (NC, T0B) per CTA: nbd, own, P (as in the launch kernel)
  const char* T1;        // (NC, T1B) per CTA: nbb, BIAS, Apar
  const __half* Wb;      // (2, C, SWB)
  const __half* Wd;      // (L, CB, SWD)
  const __half* Wc;      // (L, 2, KB CB, SWC)
  const __half* Wu;      // (L, C, SWU)
  const __half* Wsc;     // (NS, SWS)
  const __half* Wv;      // (C, SWV)
  const __half* Wo;      // (C, C) [k][c]
  const __half* W1;      // (C, FF) [k][f]
  const float* fl;       // (Fl<L>::N)
  unsigned long long* dbg;   // optional (NDBG, NC, 32) globaltimer stamps, a ring over requests
  float b2, b0, inv_n0, eps_b0, eps_b1, eps_h, eps_m;
  uint32_t seq0, tag0;   // the last request before this launch; the first expected is the next of each
  int nl, npoll;         // request-line copies; polls in flight per service warp (1, 2 or 4; <= nl)
  int gap;               // SM cycles between the first polls (the stagger)
  int classic;           // 1: lane 0 polls word 14 with an acquire load, then the payload in a second read
  int check;             // empty bodies: publish the bits' checksum (protocol tests)
  int fwd;               // 1: CTA 0 alone polls host memory and forwards the request to the other CTAs (DSMEM)
  int relax;             // 1: the result word by st.relaxed.sys (it carries all the host reads); 0: st.release.sys
  long long watchdog_ns; // 0: off
  uint8_t* xdev;         // development option, null here: a second-tier decoder's static input (M bytes, device)
  double tau;            // ... a shot with |double(c)| < tau (the host's routing test) is staged there
};

template <int L>
struct Fl {
  static constexpr int LNB = 0, BD = LNB + L * 2 * C, BC = BD + L * CB, BU = BC + L * 2 * CB, LNH = BU + L * C,
                       BV = LNH + 2 * C, U0 = BV + C, BO = U0 + D * C, LNM = BO + C, B1 = LNM + 2 * C, W2 = B1 + FF,
                       WL = W2 + FF, N = WL + 8;
  static_assert(N % 4 == 0, "16-byte multiple");
};

template <int L>
struct Lay {   // the launch kernel's layout; the stop flag in the spare word after the mbarriers
  static constexpr int bits = 0;
  static constexpr int T0 = 352;
  static constexpr int nbd = T0;
  static constexpr int own = nbd + RP * KSP * 2;
  static constexpr int P = own + 128;
  static constexpr int T1 = P + RP * C * 4;
  static constexpr int nbb = T1;
  static constexpr int BIAS = nbb + RP * NB * 2;
  static constexpr int Apar = BIAS + RP * NS * 4;
  static constexpr int A0 = Apar + RP * 8 * 4;
  static constexpr int Wb = A0 + RP * SA0 * 2;
  static constexpr int Wd = Wb + C * SWB * 2;
  static constexpr int Wc = Wd + L * CB * SWD * 2;
  static constexpr int Wu = Wc + L * KB * CB * SWC * 2;
  static constexpr int Wsc = Wu + L * C * SWU * 2;
  static constexpr int Wv = Wsc + NS * SWS * 2;
  static constexpr int u = Wv + C * SWV * 2;
  static constexpr int Y = u + L * UB;
  static constexpr int V = Y + MT * 16 * SY * 2;
  static constexpr int cp = V + MT * 16 * SV * 2;
  static constexpr int red = cp + MT * 4 * 256 * 4;
  static constexpr int G16 = red + 2 * MT * 4 * 16 * 4;
  static constexpr int V16 = G16 + MT * 16 * SG * 2;
  static constexpr int pp = V16 + MT * 16 * SY * 2;
  static constexpr int part = pp + MT * 28 * 16 * 4;
  static constexpr int gs = part + NC * C * 4;
  static constexpr int zj = gs + NC * 8 * 4;
  static constexpr int zbar = zj + 8 * 4;
  static constexpr int bar0 = zbar + 8, bar1 = zbar + 16, flag = zbar + 24, lagw = zbar + 28;
  static constexpr int Wo = zbar + 32;
  static constexpr int W1 = Wo + C * C * 2;
  static constexpr int fl = W1 + C * FF * 2;
  static constexpr int ra = fl + Fl<L>::N * 4;
  static constexpr int fwd = ra + (C + 4 * C + C + 3 * FF + 16) * 4;   // the forwarded request (16 words)
  static constexpr int total = fwd + LINE * 4;
  static_assert(nbd % 16 == 0 && nbb % 16 == 0 && own % 16 == 0 && P % 16 == 0 && BIAS % 16 == 0 &&
                    Apar % 16 == 0 && A0 % 16 == 0 && Wb % 16 == 0 && Wd % 16 == 0 && Wc % 16 == 0 &&
                    Wu % 16 == 0 && Wsc % 16 == 0 && Wv % 16 == 0 && u % 16 == 0 && Y % 16 == 0 && V % 16 == 0 &&
                    cp % 16 == 0 && red % 16 == 0 && G16 % 16 == 0 && V16 % 16 == 0 && pp % 16 == 0 &&
                    part % 16 == 0 && Wo % 16 == 0 && W1 % 16 == 0 && T1 % 16 == 0 &&
                    fl % 16 == 0 && ra % 16 == 0 && fwd % 16 == 0 && zbar % 8 == 0 && UB % 16 == 0,
                "16-byte alignment");
};

DEV unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
DEV long long clk() {
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
// system-scope accesses to mapped host memory
DEV uint32_t ld_sys(const uint32_t* a) {
  uint32_t v;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
DEV uint32_t ld_acq_sys(const uint32_t* a) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(a) : "memory");
  return v;
}
DEV void st_rel_sys64(unsigned long long* a, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
}
DEV void st_rlx_sys64(unsigned long long* a, unsigned long long v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(a), "l"(v) : "memory");
}
// the result word: seq and logit in one aligned 8-byte store, the only thing the host reads; release only when the
// check buffer zout must be ordered before it, or (development option) when a routed shot's bits were staged for a
// second-tier decoder: thread 0 of CTA 0 copies the shot's M detector bytes from its shared bits into xdev, and the
// release orders those stores before the word the host acquires before it launches that decoder
DEV void publish(const Params& p, unsigned long long w, float c, const char* bits) {
  if (p.xdev && fabs((double)c) < p.tau) {
    const uint4* src = reinterpret_cast<const uint4*>(bits);
    uint4* dst = reinterpret_cast<uint4*>(p.xdev);
#pragma unroll 3
    for (int i = 0; i < M / 16; i++) dst[i] = src[i];
    st_rel_sys64(p.res, w);
  } else if (p.relax) {
    st_rlx_sys64(p.res, w);
  } else {
    st_rel_sys64(p.res, w);
  }
}
DEV void st_rel_sys32(uint32_t* a, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(a), "r"(v) : "memory");
}
// the forwarded request: CTA 0 stores into another CTA's slot, the owner polls its own (cluster scope, per-word tags)
DEV void st_fwd(uint32_t a, uint32_t v) {
  asm volatile("st.relaxed.cluster.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
DEV uint32_t ld_fwd(uint32_t a) {
  uint32_t v;
  asm volatile("ld.relaxed.cluster.shared::cta.u32 %0, [%1];" : "=r"(v) : "r"(a) : "memory");
  return v;
}
DEV void nb_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV void nb_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEV void csync() { nb_sync(BAR_ALL, NTH); }           // the launch kernel's __syncthreads, the compute warps only
#ifdef S9_NOCLUSTER
DEV void cl_sync() { __syncthreads(); }
DEV int cl_rank() { return (int)blockIdx.x; }
#else
DEV void cl_sync() { cluster_sync(); }
DEV int cl_rank() { return (int)cluster_rank(); }
#endif
// the received bits' checksum sum_i (i + 1) bit_i (exact in fp32: at most 56,616), one warp
DEV float bits_check(const char* sm, int lane) {
  const uint8_t* bits = reinterpret_cast<const uint8_t*>(sm);
  uint32_t v = 0;
  for (int i = lane; i < M; i += 32) v += bits[i] ? (uint32_t)(i + 1) : 0u;
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return (float)v;
}

#define STAMP(ph) \
  if (p.dbg && tid == 0) p.dbg[((size_t)slot * NC + rank) * 32 + (ph)] = gtimer();

DEV void lda_t(uint32_t* r, uint32_t a0, int s, int lane) {
  ldsm4t(r, a0 + (((lane & 7) + ((lane >> 4) << 3)) * s + ((lane >> 3) & 1) * 8) * 2);
}

DEV float silu(float v) { return v / (1.0f + fexp(-v)); }
DEV float sigm(float v) { return 1.0f / (1.0f + fexp(-v)); }
DEV void gbar(int mt) { asm volatile("bar.sync %0, 128;" ::"r"(1 + mt) : "memory"); }   // the tile's four warps

// the launch kernel's ln_tile, unchanged
DEV void ln_tile(const float (&x)[2][4], const float* w, const float* bta, float eps, float* red, __half* Yt, int mt,
                 int q, int lane) {
  const int g = lane >> 2, cq = (lane & 3) * 2;
  float s0 = x[0][0] + x[0][1] + x[1][0] + x[1][1], s1 = x[0][2] + x[0][3] + x[1][2] + x[1][3];
  s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
  s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
  s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
  s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
  const float w0m = s0 * (1.f / 16), w1m = s1 * (1.f / 16);
  float d, v0 = 0.f, v1 = 0.f;
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    d = x[nt][0] - w0m; v0 += d * d;
    d = x[nt][1] - w0m; v0 += d * d;
    d = x[nt][2] - w1m; v1 += d * d;
    d = x[nt][3] - w1m; v1 += d * d;
  }
  v0 += __shfl_xor_sync(0xffffffffu, v0, 1);
  v0 += __shfl_xor_sync(0xffffffffu, v0, 2);
  v1 += __shfl_xor_sync(0xffffffffu, v1, 1);
  v1 += __shfl_xor_sync(0xffffffffu, v1, 2);
  float* r1 = red + (mt * 4) * 16;
  float* r2 = red + (MT * 4 + mt * 4) * 16;
  if ((lane & 3) == 0) {
    r1[q * 16 + g] = w0m;
    r1[q * 16 + g + 8] = w1m;
    r2[q * 16 + g] = v0;
    r2[q * 16 + g + 8] = v1;
  }
  gbar(mt);
  const float a0 = r1[g], a1 = r1[16 + g], a2 = r1[32 + g], a3 = r1[48 + g];
  const float c0 = r1[g + 8], c1 = r1[24 + g], c2 = r1[40 + g], c3 = r1[56 + g];
  const float m0 = 0.25f * ((a0 + a1) + (a2 + a3)), m1 = 0.25f * ((c0 + c1) + (c2 + c3));
  const float q0 = (r2[g] + r2[16 + g]) + (r2[32 + g] + r2[48 + g]) +
                   16.f * ((a0 - m0) * (a0 - m0) + (a1 - m0) * (a1 - m0) + (a2 - m0) * (a2 - m0) + (a3 - m0) * (a3 - m0));
  const float q1 = (r2[g + 8] + r2[24 + g]) + (r2[40 + g] + r2[56 + g]) +
                   16.f * ((c0 - m1) * (c0 - m1) + (c1 - m1) * (c1 - m1) + (c2 - m1) * (c2 - m1) + (c3 - m1) * (c3 - m1));
  const float rs0 = rsqrtf(q0 * (1.f / C) + eps), rs1 = rsqrtf(q1 * (1.f / C) + eps);
#pragma unroll
  for (int nt = 0; nt < 2; nt++) {
    const int col = q * 16 + nt * 8 + cq;
    const float w0 = w[col], w1 = w[col + 1], b0 = bta[col], b1 = bta[col + 1];
    *reinterpret_cast<uint32_t*>(Yt + g * SY + col) = h2u((x[nt][0] - m0) * rs0 * w0 + b0, (x[nt][1] - m0) * rs0 * w1 + b1);
    *reinterpret_cast<uint32_t*>(Yt + (g + 8) * SY + col) =
        h2u((x[nt][2] - m1) * rs1 * w0 + b0, (x[nt][3] - m1) * rs1 * w1 + b1);
  }
  gbar(mt);
}

// the launch kernel's down_put (cluster mode), unchanged
template <int L>
DEV void down_put(const Params& p, char* sm, const float (&x)[2][4], int l, int mt, int q, int rank, int lane,
                  uint32_t S) {
  using LY = Lay<L>;
  const float* F = reinterpret_cast<const float*>(sm + LY::fl);
  __half* Yt = reinterpret_cast<__half*>(sm + LY::Y) + mt * 16 * SY;
  ln_tile(x, F + Fl<L>::LNB + l * 2 * C, F + Fl<L>::LNB + l * 2 * C + C, l == 0 ? p.eps_b0 : p.eps_b1,
          reinterpret_cast<float*>(sm + LY::red), Yt, mt, q, lane);
  if (q >= 2) return;
  float u[4] = {0.f, 0.f, 0.f, 0.f};
  {
    uint32_t a[4][4], bb[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) lda(a[kk], S + LY::Y + (mt * 16 * SY + kk * 16) * 2, SY, lane);
#pragma unroll
    for (int kk = 0; kk < 4; kk++) ldb2(bb[kk], S + LY::Wd + (l * CB * SWD + kk * 16) * 2, SWD, lane);
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
      const uint32_t bq[2] = {q == 0 ? bb[kk][0] : bb[kk][2], q == 0 ? bb[kk][1] : bb[kk][3]};
      mma16816(u, a[kk], bq);
    }
  }
  const int cq = (lane & 3) * 2, g = lane >> 2, n = q * 8 + cq;
  (void)g;
  const float c0 = F[Fl<L>::BD + l * CB + n], c1 = F[Fl<L>::BD + l * CB + n + 1];
  const uint32_t pk[2] = {h2u(silu(u[0] + c0), silu(u[1] + c1)), h2u(silu(u[2] + c0), silu(u[3] + c1))};
#pragma unroll
  for (int h = 0; h < 2; h++) {
    const uint32_t a1 = __shfl_down_sync(0xffffffffu, pk[h], 1), a2 = __shfl_down_sync(0xffffffffu, pk[h], 2);
    const uint32_t a3 = __shfl_down_sync(0xffffffffu, pk[h], 3);
    if ((lane & 3) != 0) continue;
    const uint4 v4 = make_uint4(pk[h], a1, a2, a3);
    const int site = rank * RC + mt * 16 + g + 8 * h;
    const uint32_t off = S + LY::u + l * UB + (site * CB + q * 8) * 2;
#pragma unroll
    for (int r = 0; r < NC; r++) st_cl_v4(mapa(off, r), v4);
  }
}

// one service warp: wait until a copy of the request line carries `tag` in all NW words; *v = the lane's word
// (lanes >= NW: word NW - 1). false on the watchdog. K polls in flight, staggered by p.gap cycles at the start.
template <int K>
DEV bool poll_k(const Params& p, uint32_t tag, int lane, unsigned long long t_idle, uint32_t* out) {
  const int wi = lane < NW ? lane : NW - 1;
  const uint32_t* a[K];
  uint32_t v[K];
#pragma unroll
  for (int i = 0; i < K; i++) a[i] = p.req + (i % p.nl) * LINE + wi;
#pragma unroll
  for (int i = 0; i < K; i++) {
    v[i] = ld_sys(a[i]);
    if (i + 1 < K) {
      const long long t = clk();
      while (clk() - t < p.gap) {
      }
    }
  }
  for (;;) {
#pragma unroll
    for (int i = 0; i < K; i++) {
      if (__all_sync(0xffffffffu, (v[i] >> 24) == tag)) {
        *out = v[i];
        return true;
      }
      v[i] = ld_sys(a[i]);
    }
    if (p.watchdog_ns && gtimer() - t_idle > (unsigned long long)p.watchdog_ns) return false;
  }
}

// The classic design: lane 0 polls the command word (acquire), then every lane reads its word once more; a word whose
// tag still lags is counted (status[4]; the release / acquire pair says it never happens) and re-read
DEV bool poll_classic(const Params& p, uint32_t tag, int lane, unsigned long long t_idle, uint32_t* out, int* lag) {
  const int wi = lane < NW ? lane : NW - 1;
  for (;;) {
    uint32_t v0 = 0;
    if (lane == 0) v0 = ld_acq_sys(p.req + NW - 1);
    v0 = __shfl_sync(0xffffffffu, v0, 0);
    if ((v0 >> 24) == tag) break;
    if (p.watchdog_ns && gtimer() - t_idle > (unsigned long long)p.watchdog_ns) return false;
  }
  uint32_t v = ld_sys(p.req + wi);
  while (!__all_sync(0xffffffffu, (v >> 24) == tag)) {
    (*lag)++;
    v = ld_sys(p.req + wi);
  }
  *out = v;
  return true;
}

// lane i < 14: its word's 24 detectors -> bytes 24 i .. 24 i + 23 of the shared bits (one byte per detector)
DEV void unpack(char* bits, uint32_t v, int lane) {
  if (lane < 14) {
    uint32_t* dst = reinterpret_cast<uint32_t*>(bits + 24 * lane);
#pragma unroll
    for (int c = 0; c < 6; c++) {
      const uint32_t b = v >> (4 * c);
      dst[c] = (b & 1u) | ((b >> 1) & 1u) << 8 | ((b >> 2) & 1u) << 16 | ((b >> 3) & 1u) << 24;
    }
  }
}

template <int L, int BODY>
DEV void serve(const Params& p, char* sm) {
  using LY = Lay<L>;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, g = lane >> 2, cq = (lane & 3) * 2;
  const int mt = w >> 2, q = w & 3, gt = tid & 127;    // tile, warp in the tile's group, thread in the group
  const int rank = cl_rank();
  const int type = rank * RC >= NX;
  const uint32_t S = sa(sm);
  const float* F = reinterpret_cast<const float*>(sm + LY::fl);
#ifdef S9_NOCLUSTER
  constexpr int FL_OFF = 352, LG_OFF = 356;            // the compact layout: the bits, then the two flags
#else
  constexpr int FL_OFF = LY::flag, LG_OFF = LY::lagw;
#endif
  // ---------------------------------------------------------------------------------------------- once: residency
#ifndef S9_NOCLUSTER
  if (tid == 0) {
    const uint32_t B0 = T0B + C * SWB * 2 + L * CB * SWD * 2 + Fl<L>::N * 4;   // group 0 without the bits
    const uint32_t B1 = T1B + L * KB * CB * SWC * 2 + L * C * SWU * 2 + NS * SWS * 2 + C * SWV * 2 + C * C * 2 + C * FF * 2;
    mbar_init(S + LY::bar0, 1);
    mbar_init(S + LY::bar1, 1);
    if (rank == 0) mbar_init(S + LY::zbar, NC);
    fence_mbar_init();
    mbar_expect_tx(S + LY::bar0, B0);
    bulk_g2s(S + LY::T0, p.T0 + (size_t)rank * T0B, T0B, S + LY::bar0);
    bulk_g2s(S + LY::Wb, p.Wb + type * C * SWB, C * SWB * 2, S + LY::bar0);
    bulk_g2s(S + LY::Wd, p.Wd, L * CB * SWD * 2, S + LY::bar0);
    bulk_g2s(S + LY::fl, p.fl, Fl<L>::N * 4, S + LY::bar0);
    mbar_expect_tx(S + LY::bar1, B1);
    bulk_g2s(S + LY::T1, p.T1 + (size_t)rank * T1B, T1B, S + LY::bar1);
#pragma unroll
    for (int l = 0; l < L; l++)
      bulk_g2s(S + LY::Wc + l * KB * CB * SWC * 2, p.Wc + (size_t)(l * 2 + type) * KB * CB * SWC, KB * CB * SWC * 2,
               S + LY::bar1);
    bulk_g2s(S + LY::Wu, p.Wu, L * C * SWU * 2, S + LY::bar1);
    bulk_g2s(S + LY::Wsc, p.Wsc, NS * SWS * 2, S + LY::bar1);
    bulk_g2s(S + LY::Wv, p.Wv, C * SWV * 2, S + LY::bar1);
    bulk_g2s(S + LY::Wo, p.Wo, C * C * 2, S + LY::bar1);
    bulk_g2s(S + LY::W1, p.W1, C * FF * 2, S + LY::bar1);
  }
#endif
  if (tid < 16) sm[LY::bits + M + tid] = 0;
#ifndef S9_NOCLUSTER
  if (tid < L * 8) reinterpret_cast<uint32_t*>(sm + LY::u + (tid >> 3) * UB + ZR * CB * 2)[tid & 7] = 0u;
  if (tid < LINE) reinterpret_cast<volatile uint32_t*>(sm + LY::fwd)[tid] = 0u;
#endif
  if (tid == 0) {
    *reinterpret_cast<volatile int*>(sm + FL_OFF) = 0;
    *reinterpret_cast<volatile int*>(sm + LG_OFF) = 0;
  }
  __syncthreads();                                     // the barriers are initialised
#ifndef S9_NOCLUSTER
  mbar_wait(S + LY::bar0, 0);
  mbar_wait(S + LY::bar1, 0);
#endif
  __syncthreads();
  cl_sync();                                           // every CTA resident (its barriers initialised)
  if (rank == 0 && tid == 0) st_rel_sys32(p.status, ST_READY);
  // Without a per-request cluster barrier (body 2, and the no-cluster build) nothing ties CTAs 1-6 to CTA 0's pace: one
  // could miss a request and then wait for a tag that never comes (the stop's included). They leave here; CTA 0 serves
  // alone. Bodies 0 and 1 cannot miss: CTA 0 publishes only after every CTA has passed the request's cluster barrier.
#ifdef S9_NOCLUSTER
  if (rank != 0) return;
#else
  if constexpr (BODY == 2) {
    if (rank != 0) return;
  }
#endif
  // ---------------------------------------------------------------------------------------------- requests
  uint32_t seq = p.seq0, tag = p.tag0;
  unsigned long long t_idle = gtimer();
  int it = 0, lag = 0;
  for (;; it++) {
    seq = seq + 1u == 0u ? 1u : seq + 1u;              // 0 is skipped on wrap
    tag = tag >= 255u ? 1u : tag + 1u;                 // 1..255
    const int slot = it % NDBG;
    if (w == WS) {                                     // ------------------------------ the service warp
      uint32_t v = 0;
      bool ok = true;
#ifndef S9_NOCLUSTER
      const bool fwd = p.fwd && BODY != 2;
      if (fwd && rank != 0) {                          // the forwarded copy in this CTA's shared memory
        const int wi = lane < NW ? lane : NW - 1;
        for (;;) {
          v = ld_fwd(S + LY::fwd + 4 * wi);
          if (__all_sync(0xffffffffu, (v >> 24) == tag)) break;
        }
      } else
#else
      const bool fwd = false;
#endif
      {
        if (p.classic)
          ok = poll_classic(p, tag, lane, t_idle, &v, &lag);
        else if (p.npoll >= 4)
          ok = poll_k<4>(p, tag, lane, t_idle, &v);
        else if (p.npoll == 2)
          ok = poll_k<2>(p, tag, lane, t_idle, &v);
        else
          ok = poll_k<1>(p, tag, lane, t_idle, &v);
#ifndef S9_NOCLUSTER
        if (fwd) {                                     // CTA 0: the words (or a watchdog command) -> CTAs 1-6
          const uint32_t fv = ok ? v : (tag << 24) | (lane == NW - 1 ? CMD_WATCHDOG : 0u);
          if (lane < NW) {
#pragma unroll
            for (int r = 1; r < NC; r++) st_fwd(mapa(S + LY::fwd + 4 * lane, r), fv);
          }
        }
#endif
      }
      if (p.dbg && lane == 0) p.dbg[((size_t)slot * NC + rank) * 32 + 0] = gtimer();
      const uint32_t cmd = __shfl_sync(0xffffffffu, v, NW - 1) & 0xffffffu;
      if (cmd == CMD_WATCHDOG) ok = false;
      const bool stop = !ok || cmd == CMD_STOP;
      if (stop) {
        if (lane == 0) {
          *reinterpret_cast<volatile int*>(sm + FL_OFF) = ok ? 1 : 2;
          *reinterpret_cast<volatile int*>(sm + LG_OFF) = lag;
        }
      } else {
        unpack(sm + LY::bits, v, lane);
      }
      __syncwarp();
      nb_arrive(BAR_GO, NT);                           // bits (or the stop flag) -> the compute warps
      if (stop) break;
      t_idle = gtimer();
#ifndef S9_NOCLUSTER
      if constexpr (BODY == 1) {                       // line parities of the CTA's sites -> CTA 0
        const uint8_t* bits = reinterpret_cast<const uint8_t*>(sm + LY::bits);
        const int16_t* own = reinterpret_cast<const int16_t*>(sm + LY::own);
        const float* Ap = reinterpret_cast<const float*>(sm + LY::Apar);
        float acc[D];
#pragma unroll
        for (int j = 0; j < D; j++) acc[j] = 0.f;
        for (int r = lane; r < RC; r += 32) {
          const float bit = bits[own[r]] ? 1.f : 0.f;
#pragma unroll
          for (int j = 0; j < D; j++) acc[j] += bit * Ap[r * 8 + j];
        }
#pragma unroll
        for (int j = 0; j < D; j++) acc[j] = warp_sum(acc[j]);
        if (lane < D) {
          float vv = acc[0];
#pragma unroll
          for (int j = 1; j < D; j++)
            if (lane == j) vv = acc[j];
          st_cl_u32(mapa(S + LY::gs + (rank * 8 + lane) * 4, 0), __float_as_uint(vv));
        }
#pragma unroll
        for (int l = 0; l <= L; l++) cluster_sync();   // the body's L + 1 cluster barriers
      } else
#endif
      if constexpr (BODY == 0) {
        cl_sync();
      }
      continue;
    }
    nb_sync(BAR_GO, NT);                               // ---------------------------------- the compute warps
    if (*reinterpret_cast<volatile int*>(sm + FL_OFF)) break;
    STAMP(1);
    if constexpr (BODY == 0 || BODY == 2) {
      float ck = 0.f;
      if (p.check && w == 0) ck = bits_check(sm + LY::bits, lane);
      if constexpr (BODY == 0) cl_sync();
      if (rank == 0 && tid == 0) publish(p, ((unsigned long long)seq << 32) | __float_as_uint(ck), ck, sm + LY::bits);
      continue;
    }
#ifndef S9_NOCLUSTER
    else {
      // -------------------------------------------------------------------------------------------- stem
      float x[2][4];
      {
        __half* A0 = reinterpret_cast<__half*>(sm + LY::A0);
        const int16_t* nbd = reinterpret_cast<const int16_t*>(sm + LY::nbd);
        const uint8_t* bits = reinterpret_cast<const uint8_t*>(sm + LY::bits);
        for (int gi = gt; gi < 16 * (KSP / 8); gi += 128) {
          const int r = mt * 16 + gi / (KSP / 8), k0 = (gi % (KSP / 8)) * 8;
          const uint4 iv = *reinterpret_cast<const uint4*>(nbd + r * KSP + k0);
          const uint32_t wv[4] = {iv.x, iv.y, iv.z, iv.w};
          uint32_t o[4];
#pragma unroll
          for (int k = 0; k < 4; k++)
            o[k] = (bits[wv[k] & 0xffffu] ? 0x3C00u : 0u) | ((bits[wv[k] >> 16] ? 0x3C00u : 0u) << 16);
          *reinterpret_cast<uint4*>(A0 + r * SA0 + k0) = make_uint4(o[0], o[1], o[2], o[3]);
        }
        gbar(mt);
        STAMP(15);
        zero(x);
        {
          uint32_t a[KSP / 16][4], bb[KSP / 16][4];
#pragma unroll
          for (int k = 0; k < KSP / 16; k++) lda(a[k], S + LY::A0 + (mt * 16 * SA0 + k * 16) * 2, SA0, lane);
#pragma unroll
          for (int k = 0; k < KSP / 16; k++) ldb2(bb[k], S + LY::Wb + (q * 16 * SWB + k * 16) * 2, SWB, lane);
#pragma unroll
          for (int k = 0; k < KSP / 16; k++) {
            mma16816(x[0], a[k], bb[k]);
            mma16816(x[1], a[k], bb[k] + 2);
          }
        }
        const float* Pt = reinterpret_cast<const float*>(sm + LY::P);
        const int r0 = mt * 16 + g, r1 = r0 + 8;
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
          const int col = q * 16 + nt * 8 + cq;
          const float2 p0 = *reinterpret_cast<const float2*>(Pt + r0 * C + col);
          const float2 p1 = *reinterpret_cast<const float2*>(Pt + r1 * C + col);
          x[nt][0] = p0.x + x[nt][0];
          x[nt][1] = p0.y + x[nt][1];
          x[nt][2] = p1.x + x[nt][2];
          x[nt][3] = p1.y + x[nt][3];
        }
        STAMP(16);
        down_put<L>(p, sm, x, 0, mt, q, rank, lane, S);
      }
      STAMP(2);
      // -------------------------------------------------------------------------------------------- blocks
#pragma unroll
      for (int l = 0; l < L; l++) {
        cluster_sync();
        STAMP(4 + 2 * l);
        {
          const int16_t* nbb = reinterpret_cast<const int16_t*>(sm + LY::nbb);
          const int row = mt * 16 + (lane & 15);
          const uint32_t ub = S + LY::u + l * UB + (lane >> 4) * 16;
          const uint32_t wc = S + LY::Wc + l * KB * CB * SWC * 2;
          float acc[2][4];
          zero(acc);
          {
            constexpr int NTW = (KB + 3) / 4;
            int idx[NTW];
#pragma unroll
            for (int i = 0; i < NTW; i++) idx[i] = (q + 4 * i < KB) ? nbb[row * NB + q + 4 * i] : ZR;
            uint32_t aa[NTW][4], bb[NTW][4];
#pragma unroll
            for (int i = 0; i < NTW; i++) ldsm4(aa[i], ub + idx[i] * (CB * 2));
#pragma unroll
            for (int i = 0; i < NTW; i++) ldb2(bb[i], wc + (q + 4 * i < KB ? q + 4 * i : 0) * CB * SWC * 2, SWC, lane);
#pragma unroll
            for (int i = 0; i < NTW; i++)
              if (q + 4 * i < KB) {
                mma16816(acc[0], aa[i], bb[i]);
                mma16816(acc[1], aa[i], bb[i] + 2);
              }
          }
          float* cpw = reinterpret_cast<float*>(sm + LY::cp) + (mt * 4 + q) * 256;
#pragma unroll
          for (int nt = 0; nt < 2; nt++) {
            *reinterpret_cast<float2*>(cpw + g * 16 + nt * 8 + cq) = make_float2(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<float2*>(cpw + (g + 8) * 16 + nt * 8 + cq) = make_float2(acc[nt][2], acc[nt][3]);
          }
          gbar(mt);
          {
            const float* cpt = reinterpret_cast<const float*>(sm + LY::cp) + mt * 4 * 256;
            const int rr = gt >> 3, col = (gt & 7) * 2;
            float v0 = 0.f, v1 = 0.f;
#pragma unroll
            for (int k = 0; k < 4; k++) {
              v0 += cpt[k * 256 + rr * 16 + col];
              v1 += cpt[k * 256 + rr * 16 + col + 1];
            }
            v0 = silu(v0 + F[Fl<L>::BC + (l * 2 + type) * CB + col]);
            v1 = silu(v1 + F[Fl<L>::BC + (l * 2 + type) * CB + col + 1]);
            *reinterpret_cast<uint32_t*>(reinterpret_cast<__half*>(sm + LY::V) + (mt * 16 + rr) * SV + col) = h2u(v0, v1);
          }
          gbar(mt);
          {
            uint32_t a[4], bb[4];
            lda(a, S + LY::V + mt * 16 * SV * 2, SV, lane);
            ldb2(bb, S + LY::Wu + (l * C * SWU + q * 16 * SWU) * 2, SWU, lane);
            float t0[4] = {0.f, 0.f, 0.f, 0.f}, t1[4] = {0.f, 0.f, 0.f, 0.f};
            mma16816(t0, a, bb);
            mma16816(t1, a, bb + 2);
            const int c0 = q * 16 + cq, c1 = c0 + 8;
            const float u0 = F[Fl<L>::BU + l * C + c0], u1 = F[Fl<L>::BU + l * C + c0 + 1];
            const float u2 = F[Fl<L>::BU + l * C + c1], u3 = F[Fl<L>::BU + l * C + c1 + 1];
            x[0][0] += t0[0] + u0;
            x[0][1] += t0[1] + u1;
            x[0][2] += t0[2] + u0;
            x[0][3] += t0[3] + u1;
            x[1][0] += t1[0] + u2;
            x[1][1] += t1[1] + u3;
            x[1][2] += t1[2] + u2;
            x[1][3] += t1[3] + u3;
          }
          if (l + 1 < L) down_put<L>(p, sm, x, l + 1, mt, q, rank, lane, S);
          STAMP(5 + 2 * l);
        }
      }
      // -------------------------------------------------------------------------------------------- head
      {
        __half* Yt = reinterpret_cast<__half*>(sm + LY::Y) + mt * 16 * SY;
        ln_tile(x, F + Fl<L>::LNH, F + Fl<L>::LNH + C, p.eps_h, reinterpret_cast<float*>(sm + LY::red), Yt, mt, q, lane);
        uint32_t a[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) lda(a[kk], S + LY::Y + (mt * 16 * SY + kk * 16) * 2, SY, lane);
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        float vv[2][4];
        zero(vv);
        {
          uint32_t bs[4][4], bvv[4][4];
#pragma unroll
          for (int kk = 0; kk < 4; kk++) ldb2(bs[kk], S + LY::Wsc + ((q & 2) * 8 * SWS + kk * 16) * 2, SWS, lane);
#pragma unroll
          for (int kk = 0; kk < 4; kk++) ldb2(bvv[kk], S + LY::Wv + (q * 16 * SWV + kk * 16) * 2, SWV, lane);
#pragma unroll
          for (int kk = 0; kk < 4; kk++) {
            const uint32_t bq[2] = {(q & 1) ? bs[kk][2] : bs[kk][0], (q & 1) ? bs[kk][3] : bs[kk][1]};
            mma16816(s, a[kk], bq);
            mma16816(vv[0], a[kk], bvv[kk]);
            mma16816(vv[1], a[kk], bvv[kk] + 2);
          }
        }
        const float* bias = reinterpret_cast<const float*>(sm + LY::BIAS);
        __half* G16 = reinterpret_cast<__half*>(sm + LY::G16) + mt * 16 * SG;
        __half* V16 = reinterpret_cast<__half*>(sm + LY::V16) + mt * 16 * SY;
#pragma unroll
        for (int h = 0; h < 2; h++) {
          const int lr = g + 8 * h, row = mt * 16 + lr, col = q * 8 + cq;
          const float g0 = (col & 7) != 7 ? sigm(s[h * 2] + bias[row * NS + col]) : 0.f;
          const float g1 = ((col + 1) & 7) != 7 ? sigm(s[h * 2 + 1] + bias[row * NS + col + 1]) : 0.f;
          *reinterpret_cast<uint32_t*>(G16 + lr * SG + col) = h2u(g0, g1);
#pragma unroll
          for (int nt = 0; nt < 2; nt++) {
            const int vc = q * 16 + nt * 8 + cq;
            *reinterpret_cast<uint32_t*>(V16 + lr * SY + vc) =
                h2u(vv[nt][h * 2] + F[Fl<L>::BV + vc], vv[nt][h * 2 + 1] + F[Fl<L>::BV + vc + 1]);
          }
        }
        gbar(mt);
        uint32_t aa[4], bb[4];
        lda_t(aa, S + LY::G16 + (mt * 16 * SG + (q >> 1) * 16) * 2, SG, lane);
        ldb2t(bb, S + LY::V16 + (mt * 16 * SY + q * 16) * 2, SY, lane);
        float pq[2][4];
        zero(pq);
        mma16816(pq[0], aa, bb);
        mma16816(pq[1], aa, bb + 2);
        if (g < D) {
          float* ppw = reinterpret_cast<float*>(sm + LY::pp) + ((mt * 4 + q) * D + g) * 16;
          const bool hs = q & 1;                       // rows 8 (q % 2) + j: the fragment's second half for odd q
#pragma unroll
          for (int nt = 0; nt < 2; nt++)
            *reinterpret_cast<float2*>(ppw + nt * 8 + cq) =
                make_float2(hs ? pq[nt][2] : pq[nt][0], hs ? pq[nt][3] : pq[nt][1]);
        }
      }
      csync();
      STAMP(8);
      for (int o = tid; o < D * C; o += NTH) {
        const int j = o >> 6, col = o & (C - 1), hh = col >> 4;
        const float* pp = reinterpret_cast<const float*>(sm + LY::pp) + (hh * D + j) * 16 + (col & 15);
        const float acc = (pp[0] + pp[4 * D * 16]) + pp[8 * D * 16];
        st_cl_u32(mapa(S + LY::part + (rank * C + col) * 4, j), __float_as_uint(acc));
      }
      STAMP(9);
      cluster_sync();
      STAMP(10);
      // -------------------------------------------------------------------------------------------- read-out
      float* part = reinterpret_cast<float*>(sm + LY::part);
      float* ra = reinterpret_cast<float*>(sm + LY::ra);
      float* rt = ra + C;
      float* ry = rt + 4 * C;
      float* rm = ry + C;
      float* rz = rm + 3 * FF;
      if (tid < C) {
        float s2 = 0.f;
#pragma unroll
        for (int k = 0; k < NC; k++) s2 += part[k * C + tid];
        ra[tid] = s2 * p.inv_n0;
      }
      csync();
      STAMP(11);
      if (tid < 4 * C) {
        const __half* Wo = reinterpret_cast<const __half*>(sm + LY::Wo);
        const int col = tid & (C - 1), kc = tid >> 6;
        float acc = 0.f;
#pragma unroll
        for (int k = kc * 16; k < kc * 16 + 16; k++) acc += ra[k] * __half2float(Wo[k * C + col]);
        rt[kc * C + col] = acc;
      }
      csync();
      if (tid < C)
        ry[tid] = F[Fl<L>::U0 + rank * C + tid] + ((rt[tid] + rt[C + tid]) + (rt[2 * C + tid] + rt[3 * C + tid])) +
                  F[Fl<L>::BO + tid];
      csync();
      STAMP(12);
      if (w == 0) {
        const float v0 = ry[lane], v1 = ry[lane + 32];
        const float mean = warp_sum(v0 + v1) * (1.f / C);
        const float d0 = v0 - mean, d1 = v1 - mean;
        const float rstd = rsqrtf(warp_sum(d0 * d0 + d1 * d1) * (1.f / C) + p.eps_m);
        ry[lane] = d0 * rstd * F[Fl<L>::LNM + lane] + F[Fl<L>::LNM + C + lane];
        ry[lane + 32] = d1 * rstd * F[Fl<L>::LNM + lane + 32] + F[Fl<L>::LNM + C + lane + 32];
      }
      csync();
      if (tid < 2 * FF) {
        const __half* W1 = reinterpret_cast<const __half*>(sm + LY::W1);
        const int f = tid & (FF - 1), k0 = (tid >> 7) * 32;
        float h0 = 0.f, h1 = 0.f;
#pragma unroll
        for (int k = 0; k < 32; k += 2) {
          h0 += ry[k0 + k] * __half2float(W1[(k0 + k) * FF + f]);
          h1 += ry[k0 + k + 1] * __half2float(W1[(k0 + k + 1) * FF + f]);
        }
        rm[(tid >> 7) * FF + f] = h0 + h1;
      }
      csync();
      if (tid < FF) {
        const float hd = (rm[tid] + rm[FF + tid]) + F[Fl<L>::B1 + tid];
        const float v = warp_sum(silu(hd) * F[Fl<L>::W2 + tid]);
        if (lane == 0) rz[w] = v;
      }
      csync();
      STAMP(13);
      if (tid == 0) {
        const float zv = p.b2 + ((rz[0] + rz[1]) + (rz[2] + rz[3]));
        st_cl_u32(mapa(S + LY::zj + rank * 4, 0), __float_as_uint(zv));
        mbar_arrive_remote(mapa(S + LY::zbar, 0));
      }
      if (rank == 0 && tid == 0) {
        float zz[D], gsv[D];
        mbar_wait_cluster(S + LY::zbar, (uint32_t)(it & 1));
        const float* zs = reinterpret_cast<const float*>(sm + LY::zj);
        const float* gsum = reinterpret_cast<const float*>(sm + LY::gs);
#pragma unroll
        for (int j = 0; j < D; j++) {
          zz[j] = zs[j];
          float g2 = 0.f;
#pragma unroll
          for (int k = 0; k < NC; k++) g2 += gsum[k * 8 + j];
          gsv[j] = g2;
        }
        float c = 0.f;
#pragma unroll
        for (int j = 0; j < D; j++) {
          const float sgn = 1.0f - 2.0f * (gsv[j] - 2.0f * floorf(gsv[j] * 0.5f));
          c += F[Fl<L>::WL + j] * sgn * zz[j];
        }
        c += p.b0;
        if (p.zout) {                                  // checks only: before the publication (the release orders it)
#pragma unroll
          for (int j = 0; j < D; j++) p.zout[j] = zz[j];
          p.zout[7] = c;
        }
        publish(p, ((unsigned long long)seq << 32) | __float_as_uint(c), c, sm + LY::bits);
        STAMP(14);
      }
    }
#endif
  }
  // ---------------------------------------------------------------------------------------------- leave together
  cl_sync();
  if (rank == 0 && tid == 0) {
    p.status[1] = (uint32_t)it;
    p.status[2] = seq;
    p.status[3] = tag;
    p.status[4] = (uint32_t)*reinterpret_cast<volatile int*>(sm + LG_OFF);
    st_rel_sys32(p.status, *reinterpret_cast<volatile int*>(sm + FL_OFF) == 2 ? ST_WATCHDOG : ST_STOPPED);
  }
}

}  // namespace s9

#ifndef S9_NOCLUSTER
#define S9_DIMS __cluster_dims__(7, 1, 1)
#else
#define S9_DIMS
#endif
#define S9_KERNEL(NAME, L, BODY)                                                                        \
  extern "C" __global__ void S9_DIMS __launch_bounds__(s9::NT, 1) NAME(s9::Params p) {                  \
    extern __shared__ __align__(16) char sm[];                                                           \
    s9::serve<L, BODY>(p, sm);                                                                           \
  }

#ifndef S9_NOCLUSTER
S9_KERNEL(s9_serve_l1, 1, 1)
S9_KERNEL(s9_serve_l2, 2, 1)
#endif
S9_KERNEL(s9_hs_cluster, 1, 0)
S9_KERNEL(s9_hs_cta0, 1, 2)

// the GPU's read latency: n dependent loads from mapped host memory (ld.relaxed.sys), then n from device memory
// (ld.relaxed.gpu, L2), one thread, globaltimer -> out[0], out[1] (ns in total), out[2] the chain's value,
// out[3] the SM cycles of the host chain
extern "C" __global__ void s9_lat(const uint32_t* host, const uint32_t* dev, int n, unsigned long long* out) {
  uint32_t i = 0;
  const unsigned long long t0 = s9::gtimer();
  const long long k0 = s9::clk();
  for (int k = 0; k < n; k++) i = s9::ld_sys(host + (i & 15));
  const long long k1 = s9::clk();
  const unsigned long long t1 = s9::gtimer();
  for (int k = 0; k < n; k++) {
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(dev + (i & 15)) : "memory");
    i = v;
  }
  const unsigned long long t2 = s9::gtimer();
  out[0] = t1 - t0;
  out[1] = t2 - t1;
  out[2] = i;
  out[3] = (unsigned long long)(k1 - k0);              // SM cycles of the host chain (the clock for the stagger)
}

extern "C" __global__ void s9_smem_sizes(int* out) {
#ifdef S9_NOCLUSTER
  out[0] = out[1] = 384;                               // the compact layout of the empty bodies
#else
  out[0] = s9::Lay<1>::total;
  out[1] = s9::Lay<2>::total;
#endif
}
