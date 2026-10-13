// Device building blocks of the cluster kernel (kernel.cu): tensor-core tiles (mma.sync m16n8k16, fp16
// inputs, fp32 accumulation) on shared-memory operands, LayerNorm and GELU in fp32, distributed shared memory (DSMEM)
// stores and cluster barriers, bulk async copies (cp.async.bulk) with mbarrier completion. sm_90a.
#pragma once
#include <cuda_fp16.h>
#include <stdint.h>

#define DEV __device__ __forceinline__

namespace dtc {

// ------------------------------------------------------------------------------------------ addresses, cluster
DEV uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

DEV uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

// the shared::cluster address of local shared address a in CTA `rank`
DEV uint32_t mapa(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}

DEV void st_cl_u32(uint32_t a, uint32_t v) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(a), "r"(v) : "memory");
}
DEV void st_cl_v2(uint32_t a, uint32_t x, uint32_t y) {
  asm volatile("st.shared::cluster.v2.u32 [%0], {%1, %2};" ::"r"(a), "r"(x), "r"(y) : "memory");
}
DEV void st_cl_v4(uint32_t a, uint4 v) {
  asm volatile("st.shared::cluster.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

// every thread of every CTA of the cluster; release / acquire order the DSMEM stores before it
DEV void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
DEV void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
DEV void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

// ------------------------------------------------------------------------------------------ bulk copies
DEV void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
DEV void fence_mbar_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DEV void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n\t.reg .b64 st;\n\tmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n\t}" ::"r"(bar),
               "r"(bytes)
               : "memory");
}
DEV void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!done);
}
// global -> local shared memory, `bytes` a multiple of 16, both addresses 16-B aligned, completion counted on `bar`
DEV void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
// global -> the same shared-memory offset in every CTA of ctamask, completion on each receiver's mbarrier at `bar`
DEV void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, uint16_t ctamask) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], "
      "%4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "h"(ctamask)
      : "memory");
}
DEV void fence_proxy_async_global() { asm volatile("fence.proxy.async.global;" ::: "memory"); }
// arrive (release, cluster scope) on an mbarrier of another CTA (bar: shared::cluster address from mapa)
DEV void mbar_arrive_remote(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(bar) : "memory");
}
DEV void mbar_wait_cluster(uint32_t bar, uint32_t parity) {   // acquire at cluster scope (remote producers)
  uint32_t done = 0;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, "
        "0, p;\n\t}"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!done);
}
// programmatic dependent launch (PDL): wait until the preceding grid has completed and its memory is visible / let the
// next grid launch; both are no-ops when the kernel was not launched with the programmatic-serialization attribute
DEV void griddep_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
DEV void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
DEV void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
DEV void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// ------------------------------------------------------------------------------------------ tensor cores
DEV void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEV void ldsm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
DEV void ldsm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
DEV uint32_t h2u(float lo, float hi) {
  __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&h);
}

// A fragment (16 x 16, row-major, stride s halves) at smem address a0 (row 0, col k0 already applied)
DEV void lda(uint32_t* r, uint32_t a0, int s, int lane) { ldsm4(r, a0 + ((lane & 15) * s + (lane >> 4) * 8) * 2); }
// B fragments of two n-tiles (16 n x 16 k) from [n][k] row-major storage (stride s halves) at b0 (row n0, col k0)
DEV void ldb2(uint32_t* r, uint32_t b0, int s, int lane) {
  ldsm4(r, b0 + (((lane & 7) + ((lane >> 4) << 3)) * s + ((lane >> 3) & 1) * 8) * 2);
}
// B fragments of two n-tiles from [k][n] row-major storage (V: keys x dh), transposed on load
DEV void ldb2t(uint32_t* r, uint32_t b0, int s, int lane) {
  ldsm4t(r, b0 + ((lane & 15) * s + (lane >> 4) * 8) * 2);
}

// acc[NT][4] += A (16 x K at a0, stride sa) . B^T (B: [8 NT][K] at b0, stride sb), K a multiple of 16, NT even
template <int NT>
DEV void warp_gemm(float (&acc)[NT][4], uint32_t a0, int sa_, uint32_t b0, int sb, int K, int lane) {
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    uint32_t a[4];
    lda(a, a0 + k0 * 2, sa_, lane);
#pragma unroll
    for (int j = 0; j < NT; j += 2) {
      uint32_t b[4];
      ldb2(b, b0 + (j * 8 * sb + k0) * 2, sb, lane);
      mma16816(acc[j], a, b);
      mma16816(acc[j + 1], a, b + 2);
    }
  }
}

template <int NT>
DEV void zero(float (&acc)[NT][4]) {
#pragma unroll
  for (int j = 0; j < NT; j++) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;
}

// ------------------------------------------------------------------------------------------ elementwise
DEV float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
DEV float gelu(float h) { return 0.5f * h * (1.0f + erff(h * 0.7071067811865476f)); }

// LayerNorm of one 128-wide fp32 row (one warp; lane holds columns 4 lane .. 4 lane + 3) -> fp16 row
DEV void ln_row128(const float* x, __half* y, const float* w, const float* b, float eps, int lane) {
  float4 v = reinterpret_cast<const float4*>(x)[lane];
  float mean = warp_sum(v.x + v.y + v.z + v.w) * (1.f / 128.f);
  float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
  float rstd = rsqrtf(warp_sum(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) * (1.f / 128.f) + eps);
  float4 ww = reinterpret_cast<const float4*>(w)[lane], bb = reinterpret_cast<const float4*>(b)[lane];
  uint2 o;
  o.x = h2u(a0 * rstd * ww.x + bb.x, a1 * rstd * ww.y + bb.y);
  o.y = h2u(a2 * rstd * ww.z + bb.z, a3 * rstd * ww.w + bb.w);
  reinterpret_cast<uint2*>(y)[lane] = o;
}
// the same, fp32 output (final LayerNorm of the read-out)
DEV float4 ln_row128_f(const float* x, const float* w, const float* b, float eps, int lane) {
  float4 v = reinterpret_cast<const float4*>(x)[lane];
  float mean = warp_sum(v.x + v.y + v.z + v.w) * (1.f / 128.f);
  float a0 = v.x - mean, a1 = v.y - mean, a2 = v.z - mean, a3 = v.w - mean;
  float rstd = rsqrtf(warp_sum(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3) * (1.f / 128.f) + eps);
  float4 ww = reinterpret_cast<const float4*>(w)[lane], bb = reinterpret_cast<const float4*>(b)[lane];
  return make_float4(a0 * rstd * ww.x + bb.x, a1 * rstd * ww.y + bb.y, a2 * rstd * ww.z + bb.z,
                     a3 * rstd * ww.w + bb.w);
}

DEV float fexp(float x) {  // e^x as ex2.approx(x log2 e), the form Triton's tl.exp lowers to
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x * 1.4426950408889634f));
  return y;
}

}  // namespace dtc
