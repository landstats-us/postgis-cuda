#pragma once
// GPU spatial join engine: LBVH candidate search + warp-per-pair exact refinement
// (st_intersects and area of intersection), FP64 throughout.
#include "common.h"
#include "gset.h"
#include <type_traits>
#include <memory>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <functional>
#include <atomic>

// ---------------------------------------------------------------------------------------------
// Device-side polygon set, FP64. Coordinates are stored per geometry as v - origin (exact by Sterbenz), so
// (o_b - o_a) + (v - o_b) == v - o_a exactly and vertices shared by neighbouring polygons stay bit-identical
// in every pair frame (needed for exact touch tests). Rings are closed (first == last vertex).
// Shells are CCW and holes CW (normalized at export), which the area kernel relies on.
// ---------------------------------------------------------------------------------------------
struct DSet {
  uint32_t n = 0, nr = 0; uint64_t nv = 0;
  double2* origin = nullptr;   // bbox min, FP64
  double2* ext = nullptr;      // bbox extent
  uint32_t* gro = nullptr;     // [n+1]
  uint64_t* rvo = nullptr;     // [nr+1] (64-bit: full-table layers exceed 2^32 vertices)
  double* xy = nullptr;        // [nv*2], v - origin
  float4* bb = nullptr;        // conservative float bbox (minx,miny,maxx,maxy) in world coords
  uint32_t* parent = nullptr;  // piece -> original geometry (nullptr: not subdivided)
  uint32_t* synb = nullptr;    // bit per vertex: edge v->v+1 was created by subdivision (not real boundary)
  const uint32_t* h_parent = nullptr;   // host copy of parent (owned by caller), used for batch alignment
  double* blen = nullptr;      // length of the real boundary per parent (per geometry if not subdivided), for ST_Relate
  size_t bytes = 0;
};

inline float f_down(double v) { float f = (float)v; if ((double)f > v) f = nextafterf(f, -INFINITY); return f; }
inline float f_up(double v)   { float f = (float)v; if ((double)f < v) f = nextafterf(f, INFINITY); return f; }

template <class T> static T* dalloc_copy(const std::vector<T>& h, size_t& bytes) {
  T* d; size_t b = h.size() * sizeof(T);
  CK(cudaMalloc(&d, b ? b : 1)); CK(cudaMemcpy(d, h.data(), b, cudaMemcpyHostToDevice)); bytes += b; return d;
}

inline DSet upload_set(const HostSet& h, const std::vector<uint32_t>* parent = nullptr, const std::vector<uint8_t>* syn = nullptr) {
  if (h.nr >= (1ull << 32) || h.n >= (1ull << 31)) { fprintf(stderr, "set too large (rings >= 2^32 or geoms >= 2^31)\n"); exit(1); }
  DSet d; d.n = (uint32_t)h.n; d.nr = (uint32_t)h.nr; d.nv = h.nv;
  std::vector<double2> origin(h.n), ext(h.n); std::vector<float4> bb(h.n);
  std::vector<uint32_t> gro(h.gro.begin(), h.gro.end());
  const std::vector<uint64_t>& rvo = h.rvo;
  std::unique_ptr<double[]> q(new double[h.nv * 2]);   // default-init (no 2x47 GB zeroing); filled in parallel below
#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t g = 0; g < (int64_t)h.n; g++) {
    uint64_t v0 = h.rvo[h.gro[g]], v1 = h.rvo[h.gro[g + 1]];
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (uint64_t v = v0; v < v1; v++) {
      double x = h.xy[2 * v], y = h.xy[2 * v + 1];
      x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
    }
    origin[g] = make_double2(x0, y0);
    ext[g] = make_double2(x1 - x0, y1 - y0);
    bb[g] = make_float4(f_down(x0), f_down(y0), f_up(x1), f_up(y1));
    for (uint64_t v = v0; v < v1; v++) {
      q[2 * v] = h.xy[2 * v] - x0;
      q[2 * v + 1] = h.xy[2 * v + 1] - y0;
    }
  }
  d.origin = dalloc_copy(origin, d.bytes); d.ext = dalloc_copy(ext, d.bytes);
  d.gro = dalloc_copy(gro, d.bytes); d.rvo = dalloc_copy(rvo, d.bytes); d.bb = dalloc_copy(bb, d.bytes);
  { size_t b = h.nv * 2 * sizeof(double); CK(cudaMalloc(&d.xy, b ? b : 1)); CK(cudaMemcpy(d.xy, q.get(), b, cudaMemcpyHostToDevice)); d.bytes += b; }
  if (parent) { d.parent = dalloc_copy(*parent, d.bytes); d.h_parent = parent->data(); }
  {  // real boundary length per parent (cut edges excluded)
    std::vector<double> glen(h.n, 0.0);
#pragma omp parallel for schedule(dynamic, 4096)
    for (int64_t g = 0; g < (int64_t)h.n; g++) {
      double L = 0;
      for (uint64_t r = h.gro[g]; r < h.gro[g + 1]; r++)
        for (uint64_t v = h.rvo[r]; v + 1 < h.rvo[r + 1]; v++)
          if (!syn || !(*syn)[v]) L += hypot(h.xy[2 * v + 2] - h.xy[2 * v], h.xy[2 * v + 3] - h.xy[2 * v + 1]);
      glen[g] = L;
    }
    std::vector<double> bl(parent ? (h.n ? (size_t)(*parent)[h.n - 1] + 1 : 0) : h.n, 0.0);
    for (uint64_t g = 0; g < h.n; g++) bl[parent ? (*parent)[g] : g] += glen[g];
    d.blen = dalloc_copy(bl, d.bytes);
  }
  if (syn) {
    std::vector<uint32_t> bits((h.nv + 31) / 32, 0);
    for (uint64_t v = 0; v < h.nv; v++) if ((*syn)[v]) bits[v >> 5] |= 1u << (v & 31);
    d.synb = dalloc_copy(bits, d.bytes);
  }
  return d;
}

inline void free_set(DSet& d) {
  cudaFree(d.origin); cudaFree(d.ext); cudaFree(d.gro); cudaFree(d.rvo); cudaFree(d.xy); cudaFree(d.bb);
  if (d.parent) cudaFree(d.parent);
  if (d.synb) cudaFree(d.synb);
  if (d.blen) cudaFree(d.blen);
  d = DSet();
}

// ---------------------------------------------------------------------------------------------
// LBVH (Karras 2012) over float bboxes
// ---------------------------------------------------------------------------------------------
struct Bvh {
  uint32_t n = 0;
  int* left = nullptr; int* right = nullptr; int* parent = nullptr;  // internal nodes [0,n-1); ids >= n-1 are leaves
  float4* nbb = nullptr;       // bbox for all 2n-1 nodes
  uint32_t* sorted_id = nullptr;  // leaf order -> original index
  size_t bytes = 0;
};

__device__ __forceinline__ uint32_t expand_bits16(uint32_t v) {
  v = (v | (v << 8)) & 0x00FF00FFu; v = (v | (v << 4)) & 0x0F0F0F0Fu;
  v = (v | (v << 2)) & 0x33333333u; v = (v | (v << 1)) & 0x55555555u; return v;
}

__global__ void k_bounds_prep(const float4* bb, uint32_t n, float4* cmin_max_out) {
  // per-block partial reduction of centers -> atomics on global via two passes is overkill; use simple atomics
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 b = bb[i];
  float cx = 0.5f * (b.x + b.z), cy = 0.5f * (b.y + b.w);
  // monotonic float->int ordering trick for atomicMin/Max
  auto ord = [](float f) { int i = __float_as_int(f); return i >= 0 ? i : i ^ 0x7FFFFFFF; };
  int* o = reinterpret_cast<int*>(cmin_max_out);
  atomicMin(o + 0, ord(cx)); atomicMin(o + 1, ord(cy)); atomicMax(o + 2, ord(cx)); atomicMax(o + 3, ord(cy));
}

__global__ void k_morton(const float4* bb, uint32_t n, const float4* cmm, uint32_t* keys, uint32_t* vals) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  auto unord = [](int i) { return __int_as_float(i >= 0 ? i : i ^ 0x7FFFFFFF); };
  const int* o = reinterpret_cast<const int*>(cmm);
  float minx = unord(o[0]), miny = unord(o[1]), maxx = unord(o[2]), maxy = unord(o[3]);
  float4 b = bb[i];
  float cx = 0.5f * (b.x + b.z), cy = 0.5f * (b.y + b.w);
  float sx = maxx > minx ? 1.f / (maxx - minx) : 0.f, sy = maxy > miny ? 1.f / (maxy - miny) : 0.f;
  uint32_t ix = (uint32_t)fminf(fmaxf((cx - minx) * sx * 65535.f, 0.f), 65535.f);
  uint32_t iy = (uint32_t)fminf(fmaxf((cy - miny) * sy * 65535.f, 0.f), 65535.f);
  keys[i] = (expand_bits16(ix) << 1) | expand_bits16(iy);
  vals[i] = i;
}

__device__ __forceinline__ int kdelta(const uint32_t* k, int n, int i, int j) {
  if (j < 0 || j >= n) return -1;
  uint32_t a = k[i], b = k[j];
  if (a == b) return 32 + __clz((uint32_t)(i ^ j));
  return __clz(a ^ b);
}

__global__ void k_build_internal(const uint32_t* keys, int n, int* left, int* right, int* parent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n - 1) return;
  int d = (kdelta(keys, n, i, i + 1) - kdelta(keys, n, i, i - 1)) >= 0 ? 1 : -1;
  int dmin = kdelta(keys, n, i, i - d);
  int lmax = 2;
  while (kdelta(keys, n, i, i + lmax * d) > dmin) lmax *= 2;
  int l = 0;
  for (int t = lmax / 2; t >= 1; t /= 2)
    if (kdelta(keys, n, i, i + (l + t) * d) > dmin) l += t;
  int j = i + l * d;
  int dnode = kdelta(keys, n, i, j);
  int s = 0;
  for (int div = 2;; div *= 2) {
    int t = (l + div - 1) / div;
    if (kdelta(keys, n, i, i + (s + t) * d) > dnode) s += t;
    if (t <= 1) break;
  }
  int gamma = i + s * d + (d < 0 ? d : 0);
  int lo = min(i, j), hi = max(i, j);
  int lc = (lo == gamma) ? (n - 1 + gamma) : gamma;
  int rc = (hi == gamma + 1) ? (n - 1 + gamma + 1) : gamma + 1;
  left[i] = lc; right[i] = rc; parent[lc] = i; parent[rc] = i;
}

__global__ void k_leaf_bbox(const float4* bb, const uint32_t* sorted_id, int n, float4* nbb) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  nbb[n - 1 + i] = bb[sorted_id[i]];
}

__global__ void k_refit(int n, const int* left, const int* right, const int* parent, float4* nbb, int* flags) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int node = n - 1 + i;
  for (;;) {
    int p = parent[node];
    if (p < 0) return;
    __threadfence();
    int old = atomicAdd(&flags[p], 1);
    if (old == 0) return;           // first child to arrive stops; the second merges
    __threadfence();
    float4 a = nbb[left[p]], b = nbb[right[p]];
    nbb[p] = make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
    node = p;
  }
}

inline Bvh build_bvh(const DSet& R) {
  Bvh t; t.n = R.n; int n = (int)R.n;
  CK(cudaMalloc(&t.sorted_id, n * sizeof(uint32_t)));
  uint32_t *keys, *keys_s, *vals; float4* cmm; int* flags;
  CK(cudaMalloc(&keys, n * 4)); CK(cudaMalloc(&keys_s, n * 4)); CK(cudaMalloc(&vals, n * 4)); CK(cudaMalloc(&cmm, 16));
  int init[4] = {0x7FFFFFFF, 0x7FFFFFFF, (int)0x80000000, (int)0x80000000};
  // ordered-int encoding of +/-inf: use extreme ints
  { auto ord = [](float f) { int i; memcpy(&i, &f, 4); return i >= 0 ? i : i ^ 0x7FFFFFFF; };
    init[0] = init[1] = ord(INFINITY); init[2] = init[3] = ord(-INFINITY); }
  CK(cudaMemcpy(cmm, init, 16, cudaMemcpyHostToDevice));
  int B = 256, G = (n + B - 1) / B;
  k_bounds_prep<<<G, B>>>(R.bb, n, cmm);
  k_morton<<<G, B>>>(R.bb, n, cmm, keys, vals);
  size_t tmp = 0; cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys, keys_s, vals, t.sorted_id, n);
  void* tb; CK(cudaMalloc(&tb, tmp));
  cub::DeviceRadixSort::SortPairs(tb, tmp, keys, keys_s, vals, t.sorted_id, n);
  size_t nn = 2 * (size_t)n - 1;
  CK(cudaMalloc(&t.nbb, nn * sizeof(float4)));
  CK(cudaMalloc(&t.parent, nn * sizeof(int)));
  CK(cudaMemset(t.parent, 0xFF, nn * sizeof(int)));
  CK(cudaMalloc(&t.left, std::max(n - 1, 1) * sizeof(int))); CK(cudaMalloc(&t.right, std::max(n - 1, 1) * sizeof(int)));
  CK(cudaMalloc(&flags, std::max(n - 1, 1) * sizeof(int))); CK(cudaMemset(flags, 0, std::max(n - 1, 1) * sizeof(int)));
  k_leaf_bbox<<<G, B>>>(R.bb, t.sorted_id, n, t.nbb);
  if (n > 1) {
    k_build_internal<<<G, B>>>(keys_s, n, t.left, t.right, t.parent);
    k_refit<<<G, B>>>(n, t.left, t.right, t.parent, t.nbb, flags);
  }
  CK(cudaDeviceSynchronize());
  t.bytes = nn * (16 + 4) + 2 * (size_t)n * 4 + n * 4;
  cudaFree(keys); cudaFree(keys_s); cudaFree(vals); cudaFree(cmm); cudaFree(tb); cudaFree(flags);
  return t;
}

inline void free_bvh(Bvh& t) { cudaFree(t.left); cudaFree(t.right); cudaFree(t.parent); cudaFree(t.nbb); cudaFree(t.sorted_id); t = Bvh(); }

// Traversal: thread per query. pass 0: count; pass 1: write pairs (offset from exclusive scan).
__global__ void k_traverse(const float4* qbb, uint32_t qbase, uint32_t nq, const Bvh t, uint32_t* counts, const uint32_t* offs, uint2* pairs, int write, int upper = 0) {
  uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nq) return;
  float4 b = qbb[qbase + q];
  int stack[64]; int sp = 0;
  uint32_t cnt = 0, o = write ? offs[q] : 0;
  int n = (int)t.n;
  int node = (n > 1) ? 0 : (n - 1);   // n==1: root is the single leaf (id n-1 == 0)
  auto hit = [&](int id) { float4 c = t.nbb[id]; return !(c.x > b.z || c.z < b.x || c.y > b.w || c.w < b.y); };
  if (!hit(node)) { if (!write) counts[q] = 0; return; }
  stack[sp++] = node;
  while (sp) {
    node = stack[--sp];
    if (node >= n - 1) {
      uint32_t leaf = t.sorted_id[node - (n - 1)];
      if (!upper || leaf >= qbase + q) {
        if (write) pairs[o + cnt] = make_uint2(qbase + q, leaf);
        cnt++;
      }
    } else {
      int l = t.left[node], r = t.right[node];
      if (hit(r)) stack[sp++] = r;
      if (hit(l)) stack[sp++] = l;
    }
  }
  if (!write) counts[q] = cnt;
}

// ---------------------------------------------------------------------------------------------
// Pair refinement
// ---------------------------------------------------------------------------------------------
// A polygon expressed in a common frame (origin = reference geometry's origin), compute type double.
struct PV {
  const double* xy; const uint64_t* rvo; const uint32_t* synb; uint32_t r0, r1; double shx, shy;
  __device__ __forceinline__ int syn(uint64_t i) const { return synb ? (int)((synb[i >> 5] >> (i & 31)) & 1u) : 0; }
  __device__ __forceinline__ void v(uint64_t i, double& x, double& y) const {
    x = shx + xy[2 * (size_t)i];
    y = shy + xy[2 * (size_t)i + 1];
  }
};

__device__ __forceinline__ PV make_pv(const DSet& s, uint32_t g, double2 ref) {
  PV p; double2 o = s.origin[g];
  p.xy = s.xy; p.rvo = s.rvo; p.synb = s.synb; p.r0 = s.gro[g]; p.r1 = s.gro[g + 1];
  p.shx = o.x - ref.x; p.shy = o.y - ref.y;
  return p;
}

// Compensated 2x2 determinant (Kahan, via fma). Coordinate differences are exact (Sterbenz), so the sign
// is exact for exactly collinear inputs (shared vertices / shared edges evaluate to 0).
__device__ __forceinline__ double orient(double ax, double ay, double bx, double by, double cx, double cy) {
  double a = bx - ax, b = cy - ay, c = by - ay, d = cx - ax;
  double w = c * d;
  double e = fma(-c, d, w);
  double f = fma(a, b, -w);
  return f + e;
}
__device__ __forceinline__ bool onseg(double ax, double ay, double bx, double by, double px, double py) {
  return px >= fmin(ax, bx) && px <= fmax(ax, bx) && py >= fmin(ay, by) && py <= fmax(ay, by);
}

// Side of point p w.r.t. directed edge a->b, for point classification: is p left of a->b? Points exactly on the
// edge's line are treated as displaced by a fixed infinitesimal (eps, eps^2) (simulation of simplicity), so the
// answer is consistent for an edge and its reverse (e.g. the shared cut edge of two adjacent pieces): a point on a
// shared boundary lands in exactly one side.
__device__ __forceinline__ bool left_of(double ax, double ay, double bx, double by, double px, double py) {
  double o = orient(ax, ay, bx, by, px, py);
  if (o != 0) return o > 0;
  double dy = by - ay;
  if (dy != 0) return dy < 0;       // d/dpx orient = -(by - ay)
  return bx > ax;                   // d/dpy orient =  (bx - ax)
}

// closed-segment intersection (touching counts, as in ST_Intersects)
__device__ __forceinline__ bool seg_isect(double p0x, double p0y, double p1x, double p1y, double q0x, double q0y, double q1x, double q1y) {
  double d1 = orient(q0x, q0y, q1x, q1y, p0x, p0y), d2 = orient(q0x, q0y, q1x, q1y, p1x, p1y);
  double d3 = orient(p0x, p0y, p1x, p1y, q0x, q0y), d4 = orient(p0x, p0y, p1x, p1y, q1x, q1y);
  if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
  if (d1 == 0 && onseg(q0x, q0y, q1x, q1y, p0x, p0y)) return true;
  if (d2 == 0 && onseg(q0x, q0y, q1x, q1y, p1x, p1y)) return true;
  if (d3 == 0 && onseg(p0x, p0y, p1x, p1y, q0x, q0y)) return true;
  if (d4 == 0 && onseg(p0x, p0y, p1x, p1y, q1x, q1y)) return true;
  return false;
}

struct Ov { double x0, y0, x1, y1; };   // overlap bbox of the pair in the reference frame

// Containment probe point of ring r: a vertex that starts a real (non-synthetic) edge, so it lies on the
// true boundary. Corners created by subdivision may lie outside the original polygon.
__device__ __forceinline__ bool probe_vertex(const PV& A, uint32_t r, double& x, double& y) {
  uint64_t v = A.rvo[r], e = A.rvo[r + 1];
  if (A.synb) { while (v + 1 < e && A.syn(v)) v++; if (v + 1 >= e) return false; }
  A.v(v, x, y);
  return true;
}

__device__ __forceinline__ bool edge_outside(double px, double py, double qx, double qy, const Ov& o) {
  return fmax(px, qx) < o.x0 || fmin(px, qx) > o.x1 || fmax(py, qy) < o.y0 || fmin(py, qy) > o.y1;
}

// ST_Relate evidence bits (per pair, OR-combined across pieces)
constexpr uint32_t REL_COIN_SAME = 1;   // real boundaries coincide, interiors on the same side
constexpr uint32_t REL_COIN_OPP = 2;    // real boundaries coincide, interiors on opposite sides
constexpr uint32_t REL_TOUCH = 4;       // real boundaries meet (point or segment)
constexpr uint32_t DE9IM_PT_INTERIOR = 1 | 3 << 12 | 2 << 14 | 3 << 16;   // 0FFFFF212 (point in polygon interior)
constexpr uint32_t DE9IM_PT_BOUNDARY = 1 << 2 | 3 << 12 | 2 << 14 | 3 << 16;   // F0FFFF212 (point on polygon boundary)
constexpr int MAXS = 24;   // max split points per edge
constexpr int MAXO = 8;    // max collinear-overlap intervals per edge

// ---- fast path: per-pair local edge lists in shared memory --------------------------------------
#ifndef CAP_VAL
#define CAP_VAL 128
#endif
constexpr int CAP = CAP_VAL;             // local edges per polygon per warp
__device__ unsigned long long g_fallbacks = 0;
__device__ uint32_t g_nheavy = 0;
__device__ uint32_t* g_heavy_idx = nullptr;
#ifdef PROF
__device__ uint32_t* g_pcyc; __device__ uint16_t* g_pna; __device__ uint16_t* g_pnb; __device__ uint8_t* g_phit;
#endif

// syn: edge was created by subdivision (not real boundary). cut: bit 1 / 2 = start / end vertex is a cut point
// (a rounded intersection with a cut line, ~1 ulp off the original edge).
struct Ed { double x0, y0, x1, y1; int syn; int cut; };

__device__ int collect_edges(const PV& A, const Ov& ov, Ed* out, int lane, int cap = CAP) {
  int cnt = 0;
  for (uint32_t ra = A.r0; ra < A.r1; ra++) {
    uint64_t s = A.rvo[ra], e = A.rvo[ra + 1];
    for (uint64_t base = s; base + 1 < e; base += 32) {
      uint64_t i = base + lane;
      bool v = false; Ed ed;
      if (i + 1 < e) {
        A.v(i, ed.x0, ed.y0); A.v(i + 1, ed.x1, ed.y1); ed.syn = A.syn(i);
        ed.cut = A.synb ? (A.syn(i > s ? i - 1 : e - 2) ? 1 : 0) | (A.syn(i + 2 < e ? i + 1 : s) ? 2 : 0) : 0;
        v = !edge_outside(ed.x0, ed.y0, ed.x1, ed.y1, ov);
      }
      unsigned m = __ballot_sync(0xffffffffu, v);
      int pos = cnt + __popc(m & ((1u << lane) - 1));
      if (v && pos < cap) out[pos] = ed;
      cnt += __popc(m);
    }
  }
  return cnt;
}

// inside(c) for polygon B, lane-parallel over all edges: parity of crossings of the segment from a point
// left of B's bbox (certainly outside) to c. Uses exactly the same crossing predicate as inside_from_c, so a
// c lying exactly on an edge is perturbed consistently in both (a horizontal half-open ray would not be).
__device__ bool pip_full_warp(const PV& B, double mx, double my, int lane) {
  double ox = B.shx - (double)1, oy = my;
  bool par = false;
  for (uint32_t r = B.r0; r < B.r1; r++) {
    uint64_t s = B.rvo[r], e = B.rvo[r + 1];
    for (uint64_t j = s + 1 + lane; j < e; j += 32) {
      double x0, y0, x1, y1; B.v(j - 1, x0, y0); B.v(j, x1, y1);
      if ((y0 > my) == (y1 > my) && y0 != my && y1 != my) continue;     // cheap reject: edge entirely above/below
      bool so = left_of(x0, y0, x1, y1, ox, oy), sc = left_of(x0, y0, x1, y1, mx, my);
      if (so == sc) continue;
      bool su = orient(ox, oy, mx, my, x0, y0) > 0, sv = orient(ox, oy, mx, my, x1, y1) > 0;
      if (su != sv) par = !par;
    }
  }
  return (__popc(__ballot_sync(0xffffffffu, par)) & 1) != 0;
}

// inside(x) = inside(c) XOR parity(segment c->x crossing local edges); valid when c and x lie in the ov box
__device__ __forceinline__ bool inside_from_c(const Ed* L, int cnt, double cx, double cy, bool in_c, double x, double y) {
  bool in = in_c;
  double sx0 = fmin(cx, x), sx1 = fmax(cx, x), sy0 = fmin(cy, y), sy1 = fmax(cy, y);
  for (int k = 0; k < cnt; k++) {
    Ed e = L[k];
    if (fmax(e.x0, e.x1) < sx0 || fmin(e.x0, e.x1) > sx1 || fmax(e.y0, e.y1) < sy0 || fmin(e.y0, e.y1) > sy1) continue;
    bool sc = left_of(e.x0, e.y0, e.x1, e.y1, cx, cy), sx = left_of(e.x0, e.y0, e.x1, e.y1, x, y);
    if (sc == sx) continue;
    bool su = orient(cx, cy, x, y, e.x0, e.y0) > 0, sv = orient(cx, cy, x, y, e.x1, e.y1) > 0;
    if (su != sv) in = !in;
  }
  return in;
}

// Also, for ST_Relate (l_in != nullptr, edge is real): length of this edge strictly inside the other polygon
// (l_in) and lying on its real boundary (l_on), plus coincidence evidence bits.
__device__ double edge_contrib_loc(double px, double py, double qx, double qy, const Ed* L, int cnt, double cx, double cy, bool in_c, const Ov& ov,
                                   double tol, double eps_par, bool real = false, double* l_in = nullptr, double* l_on = nullptr,
                                   uint32_t* bits = nullptr, int cut = 0, double snap = 0) {
  double dx = qx - px, dy = qy - py;
  double len2 = dx * dx + dy * dy;
  if (len2 == 0) return 0.0;
  double tol2len2 = tol * tol * len2, eps2 = eps_par * eps_par;
  double ts[MAXS]; int ns = 0;
  double ol0[MAXO], ol1[MAXO]; bool osame[MAXO]; int oidx[MAXO]; int no = 0;
  ts[ns++] = 0; ts[ns++] = 1;
  double bx0 = fmin(px, qx) - tol, bx1 = fmax(px, qx) + tol, by0 = fmin(py, qy) - tol, by1 = fmax(py, qy) + tol;
  for (int k = 0; k < cnt; k++) {
    Ed e = L[k];
    if (fmax(e.x0, e.x1) < bx0 || fmin(e.x0, e.x1) > bx1 || fmax(e.y0, e.y1) < by0 || fmin(e.y0, e.y1) > by1) continue;
    double fx = e.x1 - e.x0, fy = e.y1 - e.y0;
    double r = dx * fy - dy * fx;
    double flen2 = fx * fx + fy * fy;
    // Coincidence, decided identically from both polygons' passes: edges are coincident if either edge's endpoints
    // lie on the other's line. "On" means within `snap` (half an ulp of the world coordinates: where an intersection
    // computed in double rounds onto the vertex, which is how GEOS nodes such configurations); cut points from
    // subdivision, which are rounded, may be up to tol off the line.
    double snap2 = snap * snap;
    double c0 = orient(px, py, qx, qy, e.x0, e.y0), c1 = orient(px, py, qx, qy, e.x1, e.y1);
    bool col = (c0 * c0 <= snap2 * len2 || ((e.cut & 1) && c0 * c0 <= tol2len2)) && (c1 * c1 <= snap2 * len2 || ((e.cut & 2) && c1 * c1 <= tol2len2));
    if (!col && flen2 > 0) {
      double g0 = orient(e.x0, e.y0, e.x1, e.y1, px, py), g1 = orient(e.x0, e.y0, e.x1, e.y1, qx, qy), tf = tol * tol * flen2;
      col = (g0 * g0 <= snap2 * flen2 || ((cut & 1) && g0 * g0 <= tf)) && (g1 * g1 <= snap2 * flen2 || ((cut & 2) && g1 * g1 <= tf));
    }
    if (col) {
      double tc = ((e.x0 - px) * dx + (e.y0 - py) * dy) / len2, te = ((e.x1 - px) * dx + (e.y1 - py) * dy) / len2;
      double lo = fmax((double)0, fmin(tc, te)), hi = fmin((double)1, fmax(tc, te));
      if (hi > lo) {
        if (lo > 0 && ns < MAXS) ts[ns++] = lo;
        if (hi < 1 && ns < MAXS) ts[ns++] = hi;
        if (no < MAXO) { ol0[no] = lo; ol1[no] = hi; osame[no] = (dx * fx + dy * fy) > 0; oidx[no] = k; no++; }
      }
    } else if (r * r > eps2 * len2 * flen2) {
      // t = tn / r must lie in (0,1): test signs first, divide only for real candidates
      double wx = e.x0 - px, wy = e.y0 - py;
      double tn = wx * fy - wy * fx;
      if (r > 0 ? (tn <= 0 || tn >= r) : (tn >= 0 || tn <= r)) continue;
      double t = tn / r, u = (wx * dy - wy * dx) / r;
      double du = tol / (flen2 > 0 ? sqrt(flen2) : (double)1);
      if (u >= -du && u <= 1 + du && ns < MAXS) ts[ns++] = t;
    }
  }
  for (int a = 1; a < ns; a++) { double v = ts[a]; int b = a - 1; while (b >= 0 && ts[b] > v) { ts[b + 1] = ts[b]; b--; } ts[b + 1] = v; }
  double acc = 0.0;
  for (int k = 0; k + 1 < ns; k++) {
    double t0 = ts[k], t1 = ts[k + 1];
    if (!(t1 > t0)) continue;
    double tm = (double)0.5 * (t0 + t1);
    // Weight of this sub-segment in the boundary chain of A∩B: the product rule for indicator functions,
    // jump(1_A 1_B) = jump(1_A) * avg(1_B) + jump(1_B) * avg(1_A), gives w = (1_B(left) + 1_B(right)) / 2,
    // which is exact and symmetric in A and B for coincident edges (shared boundaries, identical polygons,
    // zero-width bridges from subdivision). Off B's boundary both sides agree and w = 1_B(m).
    double mx = px + dx * tm, my = py + dy * tm;
    bool in_ov = mx >= ov.x0 && mx <= ov.x1 && my >= ov.y0 && my <= ov.y1;
    int net = 0, first = -1, net_real = 0, n_real = 0;
    for (int o = 0; o < no; o++)
      if (tm > ol0[o] && tm < ol1[o]) {
        int sgn = osame[o] ? 1 : -1;
        net += sgn; if (first < 0) first = o;
        if (!L[oidx[o]].syn) { net_real += sgn; n_real++; }
      }
    int vin = -1;   // lazily computed: is the midpoint inside the other polygon (piece)?
    auto inside_m = [&]() { if (vin < 0) vin = (in_ov && inside_from_c(L, cnt, cx, cy, in_c, mx, my)) ? 1 : 0; return vin; };
    if (l_in && real) {
      double seg = sqrt(len2) * (t1 - t0);
      if (net_real != 0) { *l_on += seg; *bits |= (net_real > 0 ? REL_COIN_SAME : REL_COIN_OPP) | REL_TOUCH;
      }
      else if (n_real == 0 && (net == 1 || net == -1)) *l_in += 0.5 * seg;   // on a cut edge: interior of the other polygon,
                                                                              // shared with the piece across the cut
      else if (inside_m()) *l_in += seg;
    }
    double w;
    if (first < 0) {
      w = inside_m() ? 1.0 : 0.0;
    } else if (net == 1 || net == -1) {
      w = 0.5;   // one side inside B, the other outside: no classification needed
    } else if (net == 0) {
      w = inside_m() ? 1.0 : 0.0;   // zero-width: both sides equal
    } else {
      // |net| >= 2 only for inconsistent input: derive the other side from the classified one and clamp
      Ed eb = L[oidx[first]];
      bool sb = left_of(eb.x0, eb.y0, eb.x1, eb.y1, mx, my);
      bool m_left = osame[first] ? sb : !sb;
      int v = inside_m();
      int other = m_left ? v - net : v + net;       // 1_B(left) - 1_B(right) = net
      other = other < 0 ? 0 : (other > 1 ? 1 : other);
      w = 0.5 * (v + other);
    }
    if (w != 0.0) {
      // cross products relative to c (inside the overlap box): translation-invariant, far less cancellation
      double ax = px + dx * t0 - cx, ay = py + dy * t0 - cy, bx = px + dx * t1 - cx, by = py + dy * t1 - cy;
      acc += w * (double)(ax * by - ay * bx);
    }
  }
  return acc;
}

struct RefineParams { double tol_rel = 1e-12; double eps_par = 1e-7; int want_area = 1; double tol_abs = 1e-11; int want_relate = 0;
                     int snap_half_ulp = 0; };   // 1: treat vertices within half an ulp of a line as on it (closer to GEOS noding)   // tol_abs: degrees; how far a subdivision cut point
// may be from the line it was cut from (they are ~1 ulp off); all other coincidence tests are exact
// Pieces of subdivided geometries can overlap only through cut edges: such pairs carry area (flag 2) but
// the parent pair is reported only if some piece pair has real-boundary contact or containment (flag 1).

// half an ulp of the largest world coordinate of the pair
__device__ __forceinline__ double pair_snap(double2 oa, double2 ea) {
  double m = fmax(fmax(fabs(oa.x), fabs(oa.x + ea.x)), fmax(fabs(oa.y), fabs(oa.y + ea.y)));
  return 0.5 * (nextafter(m, INFINITY) - m);
}

// warp per candidate pair
__global__ void __launch_bounds__(128) k_refine(const DSet P, const DSet R, const uint2* pairs, uint32_t np, RefineParams rp, uint8_t* isect, double* area,
                                                double4* ev, uint32_t* evb) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  Ed* LA = reinterpret_cast<Ed*>(smraw) + (size_t)wib * 2 * CAP;
  Ed* LB = LA + CAP;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  uint32_t nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t p = w; p < np; p += nw) {
    uint2 pr = pairs[p];
#ifdef PROF
    long long t_start = clock64();
#endif
    uint32_t a = pr.x, b = pr.y;
    double2 oa = P.origin[a], ob = R.origin[b];
    double2 ea = P.ext[a], eb = R.ext[b];
    double sx = ob.x - oa.x, sy = ob.y - oa.y;
    double pad = 4.0 * fmax(rp.tol_abs, (double)rp.tol_rel * fmax(fmax(ea.x, ea.y), fmax(eb.x, eb.y)));
    Ov ov{fmax(0.0, sx) - pad, fmax(0.0, sy) - pad, fmin(ea.x, sx + eb.x) + pad, fmin(ea.y, sy + eb.y) + pad};
    PV A = make_pv(P, a, oa), B = make_pv(R, b, oa);
    double tol = fmax(rp.tol_abs, (double)rp.tol_rel * fmax(ov.x1 - ov.x0, ov.y1 - ov.y0));
    const double snap = rp.snap_half_ulp ? pair_snap(oa, ea) : 0.0;
    int na = collect_edges(A, ov, LA, lane), nb = collect_edges(B, ov, LB, lane);
    bool hit; double ar = 0.0;
    double4 e4 = make_double4(0, 0, 0, 0); uint32_t eb4 = 0;   // ST_Relate evidence (lane-reduced)
    if (na <= CAP && nb <= CAP) {
      __syncwarp();
      hit = false; bool hit_any = false;
      for (int i = lane; i < na && !hit; i += 32) {
        Ed ea_ = LA[i];
        double ax0 = fmin(ea_.x0, ea_.x1), ax1 = fmax(ea_.x0, ea_.x1), ay0 = fmin(ea_.y0, ea_.y1), ay1 = fmax(ea_.y0, ea_.y1);
        for (int j = 0; j < nb; j++) {
          Ed eb_ = LB[j];
          if (fmax(eb_.x0, eb_.x1) < ax0 || fmin(eb_.x0, eb_.x1) > ax1 || fmax(eb_.y0, eb_.y1) < ay0 || fmin(eb_.y0, eb_.y1) > ay1) continue;
          if (seg_isect(ea_.x0, ea_.y0, ea_.x1, ea_.y1, eb_.x0, eb_.y0, eb_.x1, eb_.y1)) {
            hit_any = true;
            if (!(ea_.syn | eb_.syn)) { hit = true; break; }
          }
        }
      }
      hit = __any_sync(0xffffffffu, hit);
      hit_any = __any_sync(0xffffffffu, hit_any || hit);
      const bool rtouch = hit;   // real boundaries meet (before containment is considered)
      double cx = (double)(ov.x0 + 0.381966 * (ov.x1 - ov.x0)), cy = (double)(ov.y0 + 0.414213 * (ov.y1 - ov.y0));
      bool inB_c = false, inA_c = false;
      if (!hit || rp.want_area || rp.want_relate) { inB_c = pip_full_warp(B, cx, cy, lane); inA_c = pip_full_warp(A, cx, cy, lane); }
      bool contain_any = false;
      if (!hit) {
        // no real boundary contact: intersect iff one polygon lies inside the other. Probe one vertex per
        // ring: a real-edge vertex decides intersects; any vertex (incl. cut corners) decides whether a
        // piece may overlap and needs its area computed.
        bool h = false, ha = false;
        for (uint32_t ra = A.r0 + lane; ra < A.r1 && !h; ra += 32) {
          double x, y; bool real = probe_vertex(A, ra, x, y);
          if (!real) A.v(A.rvo[ra], x, y);
          if (x >= ov.x0 && x <= ov.x1 && y >= ov.y0 && y <= ov.y1 && inside_from_c(LB, nb, cx, cy, inB_c, x, y)) { ha = true; h = real; }
        }
        for (uint32_t rb = B.r0 + lane; rb < B.r1 && !h; rb += 32) {
          double x, y; bool real = probe_vertex(B, rb, x, y);
          if (!real) B.v(B.rvo[rb], x, y);
          if (x >= ov.x0 && x <= ov.x1 && y >= ov.y0 && y <= ov.y1 && inside_from_c(LA, na, cx, cy, inA_c, x, y)) { ha = true; h = real; }
        }
        hit = __any_sync(0xffffffffu, h);
        contain_any = __any_sync(0xffffffffu, ha);
      }
      if ((hit || hit_any || contain_any) && (rp.want_area || rp.want_relate)) {
        double s2 = 0.0, ai = 0, ao = 0, bi = 0, bo = 0; uint32_t bits = rtouch ? REL_TOUCH : 0;
        bool rel = rp.want_relate != 0;
        for (int i = lane; i < na; i += 32) {
          Ed e = LA[i];
          s2 += edge_contrib_loc(e.x0, e.y0, e.x1, e.y1, LB, nb, cx, cy, inB_c, ov, tol, (double)rp.eps_par, !e.syn, rel ? &ai : nullptr, &ao, &bits, e.cut, snap);
        }
        for (int i = lane; i < nb; i += 32) {
          Ed e = LB[i];
          s2 += edge_contrib_loc(e.x0, e.y0, e.x1, e.y1, LA, na, cx, cy, inA_c, ov, tol, (double)rp.eps_par, !e.syn, rel ? &bi : nullptr, &bo, &bits, e.cut, snap);
        }
        for (int o = 16; o > 0; o >>= 1) {
          s2 += __shfl_down_sync(0xffffffffu, s2, o);
          ai += __shfl_down_sync(0xffffffffu, ai, o); ao += __shfl_down_sync(0xffffffffu, ao, o);
          bi += __shfl_down_sync(0xffffffffu, bi, o); bo += __shfl_down_sync(0xffffffffu, bo, o);
        }
        ar = 0.5 * s2;
        e4 = make_double4(ai, ao, bi, bo);
        eb4 = __reduce_or_sync(0xffffffffu, bits);
      } else if (rtouch) eb4 = REL_TOUCH;
    } else {
      // oversized pair: hand over to the multi-warp heavy path
      uint32_t slot = 0;
      if (lane == 0) slot = atomicAdd(&g_nheavy, 1u);
      slot = __shfl_sync(0xffffffffu, slot, 0);
      if (lane == 0) g_heavy_idx[slot] = p;
      __syncwarp();
      continue;
    }
    // 1 = real contact (decides ST_Intersects of the parents); 2 = piece overlap contributing area only
    if (lane == 0) {
      bool ev_any = e4.x != 0 || e4.y != 0 || e4.z != 0 || e4.w != 0;
      isect[p] = hit ? 1 : ((ar != 0.0 || ev_any) ? 2 : 0);
      if (rp.want_area) area[p] = ar;
      if (rp.want_relate) { ev[p] = e4; evb[p] = eb4; }
    }
#ifdef PROF
    if (lane == 0) { g_pcyc[p] = (uint32_t)(clock64() - t_start); g_pna[p] = (uint16_t)min(na, 65535); g_pnb[p] = (uint16_t)min(nb, 65535); g_phit[p] = hit; }
#endif
    __syncwarp();
  }
}

// ---- heavy path: oversized pairs, local edge lists in global memory, work split into 32-edge tiles ------
struct PairCtx { PV A, B; Ov ov; double tol, cx, cy, snap; };

__device__ __forceinline__ PairCtx setup_pair(const DSet& P, const DSet& R, uint32_t a, uint32_t b, const RefineParams& rp) {
  double2 oa = P.origin[a], ob = R.origin[b];
  double2 ea = P.ext[a], eb = R.ext[b];
  double sx = ob.x - oa.x, sy = ob.y - oa.y;
  double pad = 4.0 * fmax(rp.tol_abs, (double)rp.tol_rel * fmax(fmax(ea.x, ea.y), fmax(eb.x, eb.y)));
  PairCtx c;
  c.ov = Ov{fmax(0.0, sx) - pad, fmax(0.0, sy) - pad, fmin(ea.x, sx + eb.x) + pad, fmin(ea.y, sy + eb.y) + pad};
  c.A = make_pv(P, a, oa); c.B = make_pv(R, b, oa);
  c.tol = fmax(rp.tol_abs, (double)rp.tol_rel * fmax(c.ov.x1 - c.ov.x0, c.ov.y1 - c.ov.y0));
  c.cx = (double)(c.ov.x0 + 0.381966 * (c.ov.x1 - c.ov.x0)); c.cy = (double)(c.ov.y0 + 0.414213 * (c.ov.y1 - c.ov.y0));
  c.snap = rp.snap_half_ulp ? pair_snap(oa, ea) : 0.0;
  return c;
}

__global__ void __launch_bounds__(128) k_heavy_count(const DSet P, const DSet R, const uint2* pairs, const uint32_t* hidx, uint32_t nh,
                                                     RefineParams rp, uint32_t* na_o, uint32_t* nb_o, uint32_t* items_o) {
  const int lane = threadIdx.x & 31;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t h = w; h < nh; h += nw) {
    uint2 pr = pairs[hidx[h]];
    PairCtx c = setup_pair(P, R, pr.x, pr.y, rp);
    int na = collect_edges(c.A, c.ov, nullptr, lane, 0), nb = collect_edges(c.B, c.ov, nullptr, lane, 0);
    if (lane == 0) { na_o[h] = na; nb_o[h] = nb; items_o[h] = (na + 31) / 32 + ((rp.want_area || rp.want_relate) ? (nb + 31) / 32 : 0); }
  }
}

__global__ void __launch_bounds__(128) k_heavy_fill(const DSet P, const DSet R, const uint2* pairs, const uint32_t* hidx, uint32_t nh, RefineParams rp,
                                                    const uint32_t* aoff, const uint32_t* boff, const uint32_t* na_a, const uint32_t* nb_a,
                                                    Ed* gla, Ed* glb, uint8_t* cbits, uint8_t* isect, double* area, double4* ev, uint32_t* evb) {
  const int lane = threadIdx.x & 31;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t h = w; h < nh; h += nw) {
    uint32_t p = hidx[h]; uint2 pr = pairs[p];
    PairCtx c = setup_pair(P, R, pr.x, pr.y, rp);
    Ed* LA = gla + aoff[h]; Ed* LB = glb + boff[h];
    int na = (int)na_a[h], nb = (int)nb_a[h];
    collect_edges(c.A, c.ov, LA, lane, na); collect_edges(c.B, c.ov, LB, lane, nb);
    __syncwarp();
    bool inB = pip_full_warp(c.B, c.cx, c.cy, lane), inA = pip_full_warp(c.A, c.cx, c.cy, lane);
    bool hh = false;
    for (uint32_t ra = c.A.r0 + lane; ra < c.A.r1 && !hh; ra += 32) {
      double x, y; if (!probe_vertex(c.A, ra, x, y)) continue;
      if (x >= c.ov.x0 && x <= c.ov.x1 && y >= c.ov.y0 && y <= c.ov.y1 && inside_from_c(LB, nb, c.cx, c.cy, inB, x, y)) hh = true;
    }
    for (uint32_t rb = c.B.r0 + lane; rb < c.B.r1 && !hh; rb += 32) {
      double x, y; if (!probe_vertex(c.B, rb, x, y)) continue;
      if (x >= c.ov.x0 && x <= c.ov.x1 && y >= c.ov.y0 && y <= c.ov.y1 && inside_from_c(LA, na, c.cx, c.cy, inA, x, y)) hh = true;
    }
    hh = __any_sync(0xffffffffu, hh);
    if (lane == 0) {
      cbits[h] = (inB ? 1 : 0) | (inA ? 2 : 0); isect[p] = hh; if (rp.want_area) area[p] = 0.0;
      if (rp.want_relate) { ev[p] = make_double4(0, 0, 0, 0); evb[p] = 0; }
    }
    __syncwarp();
  }
}

__global__ void __launch_bounds__(128) k_heavy_tiles(const DSet P, const DSet R, const uint2* pairs, const uint32_t* hidx, uint32_t nh, RefineParams rp,
                                                     const uint32_t* aoff, const uint32_t* boff, const uint32_t* na_a, const uint32_t* nb_a,
                                                     const Ed* gla, const Ed* glb, const uint8_t* cbits, const uint32_t* ioff, uint32_t nitems,
                                                     uint8_t* isect, double* area, double4* ev, uint32_t* evb) {
  const int lane = threadIdx.x & 31;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  const bool rel = rp.want_relate != 0;
  for (uint32_t it = w; it < nitems; it += nw) {
    uint32_t lo = 0, hi = nh;                       // largest h with ioff[h] <= it
    while (hi - lo > 1) { uint32_t mid = (lo + hi) >> 1; if (ioff[mid] <= it) lo = mid; else hi = mid; }
    uint32_t h = lo, k = it - ioff[h];
    uint32_t p = hidx[h]; uint2 pr = pairs[p];
    PairCtx c = setup_pair(P, R, pr.x, pr.y, rp);
    const Ed* LA = gla + aoff[h]; const Ed* LB = glb + boff[h];
    int na = (int)na_a[h], nb = (int)nb_a[h];
    uint32_t cb = cbits[h];
    uint32_t ta = (na + 31) / 32;
    bool side_a = k < ta;
    uint32_t i = (side_a ? k : k - ta) * 32 + lane;
    double s2 = 0.0, li = 0, lo_ = 0; uint32_t bits = 0; bool hit = false;
    if (side_a) {
      if (i < (uint32_t)na) {
        Ed e = LA[i];
        double ax0 = fmin(e.x0, e.x1), ax1 = fmax(e.x0, e.x1), ay0 = fmin(e.y0, e.y1), ay1 = fmax(e.y0, e.y1);
        for (int j = 0; j < nb && !e.syn; j++) {
          Ed f = LB[j];
          if (fmax(f.x0, f.x1) < ax0 || fmin(f.x0, f.x1) > ax1 || fmax(f.y0, f.y1) < ay0 || fmin(f.y0, f.y1) > ay1) continue;
          if (!(e.syn | f.syn) && seg_isect(e.x0, e.y0, e.x1, e.y1, f.x0, f.y0, f.x1, f.y1)) { hit = true; break; }
        }
        if (hit) bits |= REL_TOUCH;
        if (rp.want_area || rel)
          s2 = edge_contrib_loc(e.x0, e.y0, e.x1, e.y1, LB, nb, c.cx, c.cy, (cb & 1) != 0, c.ov, c.tol, (double)rp.eps_par, !e.syn, rel ? &li : nullptr, &lo_, &bits, e.cut, c.snap);
      }
    } else if (i < (uint32_t)nb) {
      Ed e = LB[i];
      s2 = edge_contrib_loc(e.x0, e.y0, e.x1, e.y1, LA, na, c.cx, c.cy, (cb & 2) != 0, c.ov, c.tol, (double)rp.eps_par, !e.syn, rel ? &li : nullptr, &lo_, &bits, e.cut, c.snap);
    }
    if (__any_sync(0xffffffffu, hit) && lane == 0) isect[p] = 1;
    if (rp.want_area) {
      for (int o = 16; o > 0; o >>= 1) s2 += __shfl_down_sync(0xffffffffu, s2, o);
      if (lane == 0 && s2 != 0.0) atomicAdd(&area[p], 0.5 * s2);
    }
    if (rel) {
      for (int o = 16; o > 0; o >>= 1) { li += __shfl_down_sync(0xffffffffu, li, o); lo_ += __shfl_down_sync(0xffffffffu, lo_, o); }
      bits = __reduce_or_sync(0xffffffffu, bits);
      if (lane == 0) {
        if (li != 0) atomicAdd(side_a ? &ev[p].x : &ev[p].z, li);
        if (lo_ != 0) atomicAdd(side_a ? &ev[p].y : &ev[p].w, lo_);
        if (bits) atomicOr(&evb[p], bits);
      }
    }
  }
}

__global__ void k_heavy_final(const uint32_t* hidx, uint32_t nh, uint8_t* isect, const double* area, const double4* ev) {
  uint32_t h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nh) return;
  uint32_t p = hidx[h];
  bool any = (area && area[p] != 0.0) || (ev && (ev[p].x != 0 || ev[p].y != 0 || ev[p].z != 0 || ev[p].w != 0));
  if (!isect[p] && any) isect[p] = 2;
}

void run_heavy(const DSet& P, const DSet& R, const uint2* pairs, const uint32_t* hidx, uint32_t nh, RefineParams rp, uint8_t* isect, double* area,
               double4* ev, uint32_t* evb) {
  uint32_t *na, *nb, *items, *aoff, *boff, *ioff; 
  CK(cudaMalloc(&na, nh * 4)); CK(cudaMalloc(&nb, nh * 4)); CK(cudaMalloc(&items, nh * 4));
  CK(cudaMalloc(&aoff, nh * 4)); CK(cudaMalloc(&boff, nh * 4)); CK(cudaMalloc(&ioff, (nh + 1) * 4));
  int RB = 128, G = (int)std::min<uint32_t>((nh * 32 + RB - 1) / RB, 132 * 16);
  static bool hprof = getenv("HEAVY_PROF") != nullptr;
  GpuTimer ht; if (hprof) ht.start();
  k_heavy_count<<<G, RB>>>(P, R, pairs, hidx, nh, rp, na, nb, items);
  if (hprof) fprintf(stderr, "  heavy nh=%u count=%.2fms", nh, ht.stop());
  size_t tmp = 0; cub::DeviceScan::ExclusiveSum(nullptr, tmp, na, aoff, nh);
  void* tb; CK(cudaMalloc(&tb, tmp));
  cub::DeviceScan::ExclusiveSum(tb, tmp, na, aoff, nh); cub::DeviceScan::ExclusiveSum(tb, tmp, nb, boff, nh);
  cub::DeviceScan::ExclusiveSum(tb, tmp, items, ioff, nh);
  uint32_t l[3], lc[3];
  CK(cudaMemcpy(&l[0], aoff + nh - 1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lc[0], na + nh - 1, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&l[1], boff + nh - 1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lc[1], nb + nh - 1, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&l[2], ioff + nh - 1, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lc[2], items + nh - 1, 4, cudaMemcpyDeviceToHost));
  uint64_t totA = (uint64_t)l[0] + lc[0], totB = (uint64_t)l[1] + lc[1], nitems = (uint64_t)l[2] + lc[2];
  CK(cudaMemcpy(ioff + nh, &nitems, 4, cudaMemcpyHostToDevice));   // (nitems < 2^32 asserted by scan type)
  Ed *gla, *glb; uint8_t* cbits;
  CK(cudaMalloc(&gla, totA * sizeof(Ed) + 16)); CK(cudaMalloc(&glb, totB * sizeof(Ed) + 16)); CK(cudaMalloc(&cbits, nh));
  if (hprof) ht.start();
  k_heavy_fill<<<G, RB>>>(P, R, pairs, hidx, nh, rp, aoff, boff, na, nb, gla, glb, cbits, isect, area, ev, evb);
  if (hprof) fprintf(stderr, " fill=%.2fms (edges A=%llu B=%llu items=%llu)", ht.stop(), (unsigned long long)totA, (unsigned long long)totB, (unsigned long long)nitems);
  if (hprof) ht.start();
  int TG = (int)std::min<uint64_t>((nitems * 32 + RB - 1) / RB, 132 * 32);
  if (nitems) k_heavy_tiles<<<TG, RB>>>(P, R, pairs, hidx, nh, rp, aoff, boff, na, nb, gla, glb, cbits, ioff, (uint32_t)nitems, isect, area, ev, evb);
  if (rp.want_area || rp.want_relate) k_heavy_final<<<(nh + 255) / 256, 256>>>(hidx, nh, isect, rp.want_area ? area : nullptr, rp.want_relate ? ev : nullptr);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  if (hprof) fprintf(stderr, " tiles=%.2fms\n", ht.stop());
  cudaFree(na); cudaFree(nb); cudaFree(items); cudaFree(aoff); cudaFree(boff); cudaFree(ioff); cudaFree(tb); cudaFree(gla); cudaFree(glb); cudaFree(cbits);
}

// ---------------------------------------------------------------------------------------------
// Join driver: probe set P (batched) against R's BVH. For each finished sub-batch the callback gets
// cb(pairs, areas, relate, n): intersecting pairs (P index, R index), their intersection areas (nullptr if
// !want_area) and DE-9IM codes for relate(P, R) (nullptr if !want_relate).
// ---------------------------------------------------------------------------------------------
struct JoinOpts {
  uint32_t batch_q = 1u << 22;          // probe geometries per traversal batch
  uint64_t max_pairs = 128ull << 20;    // candidate pairs per refine launch (bounds GPU scratch)
  bool no_agg = false;                  // debug: report piece pairs instead of parent pairs
  // optional consumer of each finished batch while still on the device (pairs in parent ids, areas or nullptr);
  // with device_only the host copy and host callback are skipped
  std::function<void(const uint2*, const double*, uint32_t)> on_device;
  bool device_only = false;
  const std::atomic<bool>* cancel = nullptr;   // checked between batches (client went away)
  // self-join (P and R are the same set): test each unordered pair once (a <= b). Results are symmetric;
  // callers mirror rows if they need both orders.
  bool upper = false;
  // optional: drop candidate pairs (device array, compacted in place) before refinement; returns the new count
  std::function<uint32_t(uint2*, uint32_t)> prefilter;
};
struct JoinStats { double traverse_ms = 0, refine_ms = 0, compact_ms = 0, copy_ms = 0; uint64_t candidates = 0, hits = 0, heavy = 0; };

struct PinnedBuf {   // grow-only pinned host buffer (fast D2H)
  void* p = nullptr; size_t cap = 0;
  void* get(size_t n) { if (n > cap) { if (p) cudaFreeHost(p); CK(cudaHostAlloc(&p, n, cudaHostAllocDefault)); cap = n; } return p; }
  ~PinnedBuf() { if (p) cudaFreeHost(p); }
};
struct DevBuf {   // grow-only scratch buffer
  void* p = nullptr; size_t cap = 0;
  void* get(size_t n) { if (n > cap) { if (p) cudaFree(p); CK(cudaMalloc(&p, n)); cap = n; } return p; }
  ~DevBuf() { if (p) cudaFree(p); }
};

__global__ void k_parent_keys(const uint2* pairs, uint32_t n, const uint32_t* pa, const uint32_t* pb, unsigned long long* keys) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint2 p = pairs[i];
  uint32_t a = pa ? pa[p.x] : p.x, b = pb ? pb[p.y] : p.y;
  keys[i] = ((unsigned long long)a << 32) | b;
}
__global__ void k_keep(const uint8_t* f, uint32_t n, int agg, uint8_t* keep) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) keep[i] = agg ? (f[i] != 0) : (f[i] == 1);
}
__global__ void k_gather_af(const uint32_t* idx, uint32_t n, const double* a, const uint8_t* f, double* ao, uint8_t* fo) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { uint32_t j = idx[i]; if (a) ao[i] = a[j]; fo[i] = f[j]; }
}
__global__ void k_iota(uint32_t* v, uint32_t n) { uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) v[i] = i; }
__global__ void k_is_one(const uint8_t* f, uint32_t n, uint8_t* o) { uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = f[i] == 1; }

// DE-9IM from boundary evidence of two areal geometries (A = probe, B = build side). Cells II IB IE BI BB BE EI EB EE,
// 2 bits each: 0 = F, 1 = '0', 2 = '1', 3 = '2'. Boundary "outside" = total real boundary - inside - on, which
// stays correct when either polygon is split into pieces (inside/on sum across pieces, outside does not).
__device__ __forceinline__ uint32_t de9im_code(double4 e, uint32_t bits, double LA, double LB) {
  const double ta = 1e-9 * LA, tb = 1e-9 * LB;
  bool aIn = e.x > ta, aOut = (LA - e.x - e.y) > ta, bIn = e.z > tb, bOut = (LB - e.z - e.w) > tb;
  bool same = bits & REL_COIN_SAME, opp = bits & REL_COIN_OPP, touch = bits & REL_TOUCH;
  uint32_t II = (aIn || bIn || same) ? 3 : 0, IB = bIn ? 2 : 0, IE = (aOut || bIn || opp) ? 3 : 0;
  uint32_t BI = aIn ? 2 : 0, BB = (same || opp) ? 2 : (touch ? 1 : 0), BE = aOut ? 2 : 0;
  uint32_t EI = (bOut || aIn || opp) ? 3 : 0, EB = bOut ? 2 : 0, EE = 3;
  return II | IB << 2 | IE << 4 | BI << 6 | BB << 8 | BE << 10 | EI << 12 | EB << 14 | EE << 16;
}
__global__ void k_rel_codes(const uint2* pairs, uint32_t n, const double4* ev, const uint32_t* evb, const double* blA, const double* blB, uint32_t* code) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) code[i] = de9im_code(ev[i], evb[i], blA[pairs[i].x], blB[pairs[i].y]);
}
__global__ void k_gather_rel(const uint32_t* idx, uint32_t n, const double4* e, const uint32_t* b, double4* eo, uint32_t* bo) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { uint32_t j = idx[i]; eo[i] = e[j]; bo[i] = b[j]; }
}
struct Sum4 { __host__ __device__ double4 operator()(const double4& a, const double4& b) const { return make_double4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); } };
struct BitOr { __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a | b; } };
struct MinU8 { __host__ __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return a < b ? a : b; } };
struct MaxU8 { __host__ __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return a > b ? a : b; } };

// DE-9IM code -> 9-character string / transpose (relate(b, a) = transpose(relate(a, b)))
inline void de9im_str(uint32_t code, char* out) { for (int k = 0; k < 9; k++) out[k] = "F012"[(code >> (2 * k)) & 3]; }
inline uint32_t de9im_transpose(uint32_t c) {
  uint32_t t = 0;
  for (int r = 0; r < 3; r++) for (int q = 0; q < 3; q++) t |= ((c >> (2 * (3 * r + q))) & 3) << (2 * (3 * q + r));
  return t;
}
// DE-9IM pattern match (chars T F * 0 1 2, case-insensitive)
inline bool de9im_match(uint32_t code, const char* pat) {
  for (int k = 0; k < 9; k++) {
    char p = pat[k]; uint32_t v = (code >> (2 * k)) & 3;
    if (p == '*') continue;
    if (p == 'T' || p == 't') { if (v == 0) return false; continue; }
    if (p == 'F' || p == 'f') { if (v != 0) return false; continue; }
    if (p >= '0' && p <= '2') { if (v != (uint32_t)(p - '0') + 1) return false; continue; }
    return false;
  }
  return true;
}

__global__ void k_unpack_keys(const unsigned long long* keys, uint32_t n, uint2* pairs) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) pairs[i] = make_uint2((uint32_t)(keys[i] >> 32), (uint32_t)keys[i]);
}

template <class F>
JoinStats join_stream(const DSet& P, const DSet& R, const Bvh& bvh, RefineParams rp, JoinOpts o, F&& cb) {
  JoinStats st;
  if (P.n == 0 || R.n == 0) return st;
  DevBuf b_counts, b_offs, b_pairs, b_is, b_ar, b_heavy, b_tmp, b_opairs, b_oar, b_nsel, b_keys, b_keys2, b_ar2, b_keep, b_oflag, b_idx, b_idx2, b_f2;
  DevBuf b_ev, b_evb, b_oev, b_oevb, b_ev2, b_evb2, b_code;
  const bool rel = rp.want_relate != 0;
  const bool agg = (P.parent || R.parent) && !o.no_agg;
  const uint32_t* hp = P.h_parent;   // pieces of one parent must stay in one sub-batch
  std::vector<uint32_t> hcounts;
  PinnedBuf h_pairs, h_area, h_code;
  size_t smem = (size_t)(128 / 32) * 2 * CAP * sizeof(Ed);
  CK(cudaFuncSetAttribute(k_refine, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  GpuTimer tm;
  for (uint32_t q0 = 0, qn; q0 < P.n; q0 = qn) {
    qn = std::min<uint32_t>(q0 + o.batch_q, P.n);
    if (hp) while (qn < P.n && hp[qn] == hp[qn - 1]) qn++;
    if (o.cancel && o.cancel->load()) break;
    uint32_t nq = qn - q0;
    uint32_t* counts = (uint32_t*)b_counts.get((size_t)nq * 4);
    uint32_t* offs = (uint32_t*)b_offs.get((size_t)nq * 4);
    tm.start();
    k_traverse<<<(nq + 127) / 128, 128>>>(P.bb, q0, nq, bvh, counts, nullptr, nullptr, 0, o.upper ? 1 : 0);
    hcounts.resize(nq);
    CK(cudaMemcpy(hcounts.data(), counts, (size_t)nq * 4, cudaMemcpyDeviceToHost));
    st.traverse_ms += tm.stop();
    // split the batch into sub-ranges whose candidate count fits max_pairs
    uint32_t s0 = 0;
    while (s0 < nq && !(o.cancel && o.cancel->load())) {
      uint64_t np = 0; uint32_t s1 = s0;
      while (s1 < nq && (np + hcounts[s1] <= o.max_pairs || s1 == s0 || (hp && hp[q0 + s1] == hp[q0 + s1 - 1]))) np += hcounts[s1++];
      uint32_t ns = s1 - s0;
      if (np >= (1ull << 32)) { fprintf(stderr, "a single probe geometry has >= 2^32 candidates\n"); exit(1); }
      tm.start();
      size_t tmp = 0; cub::DeviceScan::ExclusiveSum(nullptr, tmp, counts + s0, offs, ns);
      cub::DeviceScan::ExclusiveSum(b_tmp.get(tmp), tmp, counts + s0, offs, ns);
      uint2* pairs = (uint2*)b_pairs.get(np * sizeof(uint2) + 8);
      k_traverse<<<(ns + 127) / 128, 128>>>(P.bb, q0 + s0, ns, bvh, counts + s0, offs, pairs, 1, o.upper ? 1 : 0);
      st.traverse_ms += tm.stop();
      st.candidates += np;
      if (o.prefilter && np) np = o.prefilter(pairs, (uint32_t)np);
      if (np) {
        uint8_t* d_is = (uint8_t*)b_is.get(np + 1);
        double* d_ar = rp.want_area ? (double*)b_ar.get(np * 8 + 8) : nullptr;
        double4* d_ev = rel ? (double4*)b_ev.get(np * sizeof(double4) + 32) : nullptr;
        uint32_t* d_evb = rel ? (uint32_t*)b_evb.get(np * 4 + 4) : nullptr;
        uint32_t* d_heavy = (uint32_t*)b_heavy.get(np * 4 + 4);
        uint32_t zero32 = 0;
        CK(cudaMemcpyToSymbol(g_nheavy, &zero32, 4)); CK(cudaMemcpyToSymbol(g_heavy_idx, &d_heavy, 8));
        tm.start();
        int RB = 128, RG = (int)std::min<uint64_t>((np * 32 + RB - 1) / RB, 132 * 16);
#ifdef PROF
        uint32_t* pc; uint16_t *px, *py; uint8_t* ph;
        CK(cudaMalloc(&pc, np * 4 + 4)); CK(cudaMalloc(&px, np * 2 + 2)); CK(cudaMalloc(&py, np * 2 + 2)); CK(cudaMalloc(&ph, np + 1));
        CK(cudaMemcpyToSymbol(g_pcyc, &pc, 8)); CK(cudaMemcpyToSymbol(g_pna, &px, 8)); CK(cudaMemcpyToSymbol(g_pnb, &py, 8)); CK(cudaMemcpyToSymbol(g_phit, &ph, 8));
#endif
        k_refine<<<RG, RB, smem>>>(P, R, pairs, (uint32_t)np, rp, d_is, d_ar, d_ev, d_evb);
#ifdef PROF
        { CK(cudaDeviceSynchronize()); std::vector<uint32_t> c(np); std::vector<uint16_t> x(np), y(np); std::vector<uint8_t> hh(np);
          CK(cudaMemcpy(c.data(), pc, np * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(x.data(), px, np * 2, cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(y.data(), py, np * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hh.data(), ph, np, cudaMemcpyDeviceToHost));
          FILE* f = fopen("data/prof.bin", "wb"); fwrite(&np, 8, 1, f); fwrite(c.data(), 4, np, f); fwrite(x.data(), 2, np, f); fwrite(y.data(), 2, np, f); fwrite(hh.data(), 1, np, f); fclose(f);
          cudaFree(pc); cudaFree(px); cudaFree(py); cudaFree(ph); }
#endif
        CK(cudaGetLastError());
        uint32_t nheavy = 0; CK(cudaMemcpyFromSymbol(&nheavy, g_nheavy, 4));
        if (nheavy) run_heavy(P, R, pairs, d_heavy, nheavy, rp, d_is, d_ar, d_ev, d_evb);
        st.refine_ms += tm.stop(); st.heavy += nheavy;
        // compact on device. Non-aggregated: keep flag 1. Aggregated (pieces): keep flags 1|2, fold into
        // (parent a, parent b) with summed area and min(flag), then keep parents whose flag is 1.
        tm.start();
        uint2* opairs = (uint2*)b_opairs.get(np * sizeof(uint2) + 8);
        double* oar = rp.want_area ? (double*)b_oar.get(np * 8 + 8) : nullptr;
        int* nsel = (int*)b_nsel.get(8);
        uint8_t* keep = (uint8_t*)b_keep.get(np + 1);
        uint8_t* oflag = (uint8_t*)b_oflag.get(np + 1);
        k_keep<<<(uint32_t)((np + 255) / 256), 256>>>(d_is, (uint32_t)np, (agg || o.no_agg) ? 1 : 0, keep);
        tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, pairs, keep, opairs, nsel, (int)np);
        cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, pairs, keep, opairs, nsel, (int)np);
        int nh = 0; CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
        if (rp.want_area) {
          tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, d_ar, keep, oar, nsel, (int)np);
          cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, d_ar, keep, oar, nsel, (int)np);
        }
        double4* oev = rel ? (double4*)b_oev.get(np * sizeof(double4) + 32) : nullptr;
        uint32_t* oevb = rel ? (uint32_t*)b_oevb.get(np * 4 + 4) : nullptr;
        if (rel) {
          tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, d_ev, keep, oev, nsel, (int)np);
          cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, d_ev, keep, oev, nsel, (int)np);
          tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, d_evb, keep, oevb, nsel, (int)np);
          cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, d_evb, keep, oevb, nsel, (int)np);
        }
        if (agg && nh) {
          tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, d_is, keep, oflag, nsel, (int)np);
          cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, d_is, keep, oflag, nsel, (int)np);
          unsigned long long* keys = (unsigned long long*)b_keys.get((size_t)nh * 8);
          unsigned long long* keys2 = (unsigned long long*)b_keys2.get((size_t)nh * 8);
          uint32_t* idx = (uint32_t*)b_idx.get((size_t)nh * 4);
          uint32_t* idx2 = (uint32_t*)b_idx2.get((size_t)nh * 4);
          double* ar2 = rp.want_area ? (double*)b_ar2.get((size_t)nh * 8) : nullptr;
          uint8_t* f2 = (uint8_t*)b_f2.get((size_t)nh + 1);
          uint32_t G = (nh + 255) / 256;
          k_parent_keys<<<G, 256>>>(opairs, nh, P.parent, R.parent, keys);
          k_iota<<<G, 256>>>(idx, nh);
          tmp = 0; cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys, keys2, idx, idx2, nh);
          cub::DeviceRadixSort::SortPairs(b_tmp.get(tmp), tmp, keys, keys2, idx, idx2, nh);
          k_gather_af<<<G, 256>>>(idx2, nh, oar, oflag, ar2, f2);
          if (rp.want_area) {
            tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, keys2, keys, ar2, oar, nsel, ::cuda::std::plus<double>{}, nh);
            cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, keys2, keys, ar2, oar, nsel, ::cuda::std::plus<double>{}, nh);
          }
          double4* ev2 = rel ? (double4*)b_ev2.get((size_t)nh * sizeof(double4) + 32) : nullptr;
          uint32_t* evb2 = rel ? (uint32_t*)b_evb2.get((size_t)nh * 4 + 4) : nullptr;
          if (rel) {   // sum boundary lengths, OR evidence bits per parent pair
            k_gather_rel<<<G, 256>>>(idx2, nh, oev, oevb, ev2, evb2);
            tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, keys2, keys, ev2, oev, nsel, Sum4{}, nh);
            cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, keys2, keys, ev2, oev, nsel, Sum4{}, nh);
            tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, keys2, keys, evb2, oevb, nsel, BitOr{}, nh);
            cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, keys2, keys, evb2, oevb, nsel, BitOr{}, nh);
          }
          tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, keys2, keys, f2, oflag, nsel, MinU8{}, nh);
          cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, keys2, keys, f2, oflag, nsel, MinU8{}, nh);
          int nr = 0; CK(cudaMemcpy(&nr, nsel, 4, cudaMemcpyDeviceToHost));
          // keep parent pairs with real contact
          k_is_one<<<(nr + 255) / 256, 256>>>(oflag, nr, f2);
          tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, keys, f2, keys2, nsel, nr);
          cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, keys, f2, keys2, nsel, nr);
          if (rp.want_area) {
            tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, oar, f2, ar2, nsel, nr);
            cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, oar, f2, ar2, nsel, nr);
            CK(cudaMemcpyAsync(oar, ar2, (size_t)nr * 8, cudaMemcpyDeviceToDevice));
          }
          if (rel) {
            tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, oev, f2, ev2, nsel, nr);
            cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, oev, f2, ev2, nsel, nr);
            tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, oevb, f2, evb2, nsel, nr);
            cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, oevb, f2, evb2, nsel, nr);
            CK(cudaMemcpyAsync(oev, ev2, (size_t)nr * sizeof(double4), cudaMemcpyDeviceToDevice));
            CK(cudaMemcpyAsync(oevb, evb2, (size_t)nr * 4, cudaMemcpyDeviceToDevice));
          }
          CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
          k_unpack_keys<<<(nh + 255) / 256, 256>>>(keys2, nh, opairs);
        }
        uint32_t* dcode = nullptr;
        if (rel && nh) {
          dcode = (uint32_t*)b_code.get((size_t)nh * 4 + 4);
          k_rel_codes<<<(nh + 255) / 256, 256>>>(opairs, nh, oev, oevb, P.blen, R.blen, dcode);
        }
        st.compact_ms += tm.stop();
        if (o.on_device && nh) o.on_device(opairs, rp.want_area ? oar : nullptr, (uint32_t)nh);
        if (o.device_only) { st.hits += nh; s0 = s1; continue; }
        tm.start();
        uint2* hp_ = (uint2*)h_pairs.get((size_t)nh * sizeof(uint2) + 8);
        double* ha_ = rp.want_area ? (double*)h_area.get((size_t)nh * 8 + 8) : nullptr;
        uint32_t* hc_ = rel ? (uint32_t*)h_code.get((size_t)nh * 4 + 4) : nullptr;
        CK(cudaMemcpy(hp_, opairs, (size_t)nh * sizeof(uint2), cudaMemcpyDeviceToHost));
        if (rp.want_area) CK(cudaMemcpy(ha_, oar, (size_t)nh * 8, cudaMemcpyDeviceToHost));
        if (rel && nh) CK(cudaMemcpy(hc_, dcode, (size_t)nh * 4, cudaMemcpyDeviceToHost));
        st.copy_ms += tm.stop();
        st.hits += nh;
        if (nh) cb(hp_, ha_, hc_, (size_t)nh);
      }
      s0 = s1;
    }
  }
  return st;
}

// ---------------------------------------------------------------------------------------------
// Points x polygons: ST_Intersects(point, polygon) (boundary counts). Thread per candidate pair.
// ---------------------------------------------------------------------------------------------
struct PSetD { uint32_t n = 0; double2* xy = nullptr; float4* bb = nullptr; size_t bytes = 0; };

inline PSetD upload_points(const std::vector<double>& xy) {
  PSetD d; d.n = (uint32_t)(xy.size() / 2);
  std::vector<float4> bb(d.n);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < (int64_t)d.n; i++) bb[i] = make_float4(f_down(xy[2 * i]), f_down(xy[2 * i + 1]), f_up(xy[2 * i]), f_up(xy[2 * i + 1]));
  std::vector<double2> p(d.n); for (uint32_t i = 0; i < d.n; i++) p[i] = make_double2(xy[2 * i], xy[2 * i + 1]);
  d.xy = dalloc_copy(p, d.bytes); d.bb = dalloc_copy(bb, d.bytes);
  return d;
}
inline void free_points(PSetD& d) { cudaFree(d.xy); cudaFree(d.bb); d = PSetD(); }

__global__ void k_pip_pairs(const PSetD pts, const DSet R, const uint2* pairs, uint32_t np, uint8_t* hit) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  uint2 pr = pairs[i];
  double2 pt = pts.xy[pr.x], o = R.origin[pr.y];
  PV B = make_pv(R, pr.y, o);
  double px = (double)(pt.x - o.x), py = (double)(pt.y - o.y);          // exact for FP64 (Sterbenz)
  double ox = B.shx - (double)1;                            // left of the bbox: outside
  bool par = false, on = false;
  for (uint32_t r = B.r0; r < B.r1 && !on; r++) {
    uint64_t s0 = B.rvo[r], e = B.rvo[r + 1];
    double x0, y0; B.v(s0, x0, y0);
    for (uint64_t j = s0 + 1; j < e; j++) {
      double x1, y1; B.v(j, x1, y1);
      if (!((y0 > py) == (y1 > py) && y0 != py && y1 != py)) {
        double o_ = orient(x0, y0, x1, y1, px, py);
        if (o_ == 0 && !B.syn(j - 1) && onseg(x0, y0, x1, y1, px, py)) { on = true; break; }   // on a real edge
        bool so = left_of(x0, y0, x1, y1, ox, py), sp = left_of(x0, y0, x1, y1, px, py);
        if (so != sp) {
          bool su = orient(ox, py, px, py, x0, y0) > 0, sv = orient(ox, py, px, py, x1, y1) > 0;
          if (su != sv) par = !par;
        }
      }
      x0 = x1; y0 = y1;
    }
  }
  hit[i] = on ? 2 : (par ? 1 : 0);   // 1 = in the polygon's interior, 2 = on its boundary
}

struct PointJoinStats { double traverse_ms = 0, refine_ms = 0, compact_ms = 0, copy_ms = 0; uint64_t candidates = 0, hits = 0; };

// callback(const uint2* pairs /* (point index, parent polygon index) */, const uint8_t* where /* 1 interior, 2 boundary */, size_t n)
// DE-9IM of (point, polygon): interior -> 0FFFFF212, boundary -> F0FFFF212.
template <class F>
PointJoinStats join_points_stream(const PSetD& pts, const DSet& R, const Bvh& bvh, JoinOpts o, F&& cb) {
  PointJoinStats st;
  if (pts.n == 0 || R.n == 0) return st;
  DevBuf b_counts, b_offs, b_pairs, b_hit, b_tmp, b_opairs, b_nsel, b_keys, b_keys2, b_ohit, b_ohit2;
  std::vector<uint32_t> hcounts; PinnedBuf h_pairs, h_where;
  GpuTimer tm;
  for (uint32_t q0 = 0; q0 < pts.n; q0 += o.batch_q) {
    uint32_t nq = std::min<uint32_t>(o.batch_q, pts.n - q0);
    if (o.cancel && o.cancel->load()) break;
    uint32_t* counts = (uint32_t*)b_counts.get((size_t)nq * 4);
    uint32_t* offs = (uint32_t*)b_offs.get((size_t)nq * 4);
    tm.start();
    k_traverse<<<(nq + 127) / 128, 128>>>(pts.bb, q0, nq, bvh, counts, nullptr, nullptr, 0);
    hcounts.resize(nq); CK(cudaMemcpy(hcounts.data(), counts, (size_t)nq * 4, cudaMemcpyDeviceToHost));
    st.traverse_ms += tm.stop();
    for (uint32_t s0 = 0; s0 < nq;) {
      uint64_t np = 0; uint32_t s1 = s0;
      while (s1 < nq && (np + hcounts[s1] <= o.max_pairs || s1 == s0)) np += hcounts[s1++];
      uint32_t ns = s1 - s0;
      tm.start();
      size_t tmp = 0; cub::DeviceScan::ExclusiveSum(nullptr, tmp, counts + s0, offs, ns);
      cub::DeviceScan::ExclusiveSum(b_tmp.get(tmp), tmp, counts + s0, offs, ns);
      uint2* pairs = (uint2*)b_pairs.get(np * sizeof(uint2) + 8);
      k_traverse<<<(ns + 127) / 128, 128>>>(pts.bb, q0 + s0, ns, bvh, counts + s0, offs, pairs, 1);
      st.traverse_ms += tm.stop(); st.candidates += np;
      if (np) {
        uint8_t* hit = (uint8_t*)b_hit.get(np + 1);
        tm.start();
        k_pip_pairs<<<(uint32_t)((np + 127) / 128), 128>>>(pts, R, pairs, (uint32_t)np, hit);
        CK(cudaGetLastError());
        st.refine_ms += tm.stop();
        tm.start();
        uint2* opairs = (uint2*)b_opairs.get(np * sizeof(uint2) + 8);
        int* nsel = (int*)b_nsel.get(8);
        tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, pairs, hit, opairs, nsel, (int)np);
        cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, pairs, hit, opairs, nsel, (int)np);
        int nh = 0; CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
        uint8_t* ohit = (uint8_t*)b_ohit.get(np + 1);
        tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, hit, hit, ohit, nsel, (int)np);
        cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, hit, hit, ohit, nsel, (int)np);
        if (R.parent && nh) {   // pieces -> parent polygon, deduplicated (boundary wins over interior)
          unsigned long long* keys = (unsigned long long*)b_keys.get((size_t)nh * 8);
          unsigned long long* keys2 = (unsigned long long*)b_keys2.get((size_t)nh * 8);
          uint8_t* ohit2 = (uint8_t*)b_ohit2.get((size_t)nh + 1);
          k_parent_keys<<<(nh + 255) / 256, 256>>>(opairs, nh, nullptr, R.parent, keys);
          tmp = 0; cub::DeviceRadixSort::SortPairs(nullptr, tmp, keys, keys2, ohit, ohit2, nh);
          cub::DeviceRadixSort::SortPairs(b_tmp.get(tmp), tmp, keys, keys2, ohit, ohit2, nh);
          tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, keys2, keys, ohit2, ohit, nsel, MaxU8{}, nh);
          cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, keys2, keys, ohit2, ohit, nsel, MaxU8{}, nh);
          CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
          k_unpack_keys<<<(nh + 255) / 256, 256>>>(keys, nh, opairs);
        }
        st.compact_ms += tm.stop();
        tm.start();
        uint2* hp_ = (uint2*)h_pairs.get((size_t)nh * sizeof(uint2) + 8);
        uint8_t* hw_ = (uint8_t*)h_where.get((size_t)nh + 1);
        CK(cudaMemcpy(hp_, opairs, (size_t)nh * sizeof(uint2), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hw_, ohit, (size_t)nh, cudaMemcpyDeviceToHost));
        st.copy_ms += tm.stop(); st.hits += nh;
        if (nh) cb(hp_, hw_, (size_t)nh);
      }
      s0 = s1;
    }
  }
  return st;
}

// ---------------------------------------------------------------------------------------------
// ST_ClusterIntersecting: connected components of the intersects graph (lock-free union-find)
// ---------------------------------------------------------------------------------------------
// parent[] is read with __ldcg (L2, coherent across SMs): L1 is not coherent with atomics issued by other SMs
__device__ __forceinline__ uint32_t uf_find(uint32_t* parent, uint32_t x) {
  for (;;) {
    uint32_t p = __ldcg(parent + x);
    if (p == x) return x;
    uint32_t gp = __ldcg(parent + p);
    if (gp != p) __stcg(parent + x, gp);   // path halving (benign race: always an ancestor)
    x = p;
  }
}
__global__ void k_uf_init(uint32_t* parent, uint32_t n) { uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) parent[i] = i; }
__global__ void k_uf_union(const uint2* pairs, uint32_t n, uint32_t* parent) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t a = pairs[i].x, b = pairs[i].y;
  if (a == b) return;
  for (;;) {
    uint32_t ra = uf_find(parent, a), rb = uf_find(parent, b);
    if (ra == rb) return;
    if (ra > rb) { uint32_t t = ra; ra = rb; rb = t; }
    if (atomicCAS(&parent[rb], rb, ra) == rb) return;   // hook larger root under smaller
  }
}
// Roots go to a separate array: writing them into parent[] would race with other threads' path-halving stores,
// which can replace an already-written root with an older ancestor.
__global__ void k_uf_roots(uint32_t* parent, uint32_t n, uint32_t* root) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) root[i] = uf_find(parent, i);
}

struct ClusterStats { JoinStats join; double uf_ms = 0; };

// root[g] for every parent geometry g (root = smallest index in its component)
std::vector<uint32_t> cluster_intersecting(const DSet& P, uint32_t nparent, const Bvh& bvh, RefineParams rp, JoinOpts o, ClusterStats* cs) {
  uint32_t* parent; CK(cudaMalloc(&parent, (size_t)nparent * 4));
  k_uf_init<<<(nparent + 255) / 256, 256>>>(parent, nparent);
  double uf_ms = 0;
  rp.want_area = 0;
  o.device_only = true;
  o.upper = true;   // intersects is symmetric: each unordered pair once
  o.on_device = [&](const uint2* d_pairs, const double*, uint32_t nh) {
    GpuTimer t; t.start(); k_uf_union<<<(nh + 255) / 256, 256>>>(d_pairs, nh, parent); uf_ms += t.stop();
  };
  JoinStats js = join_stream(P, P, bvh, rp, o, [](const uint2*, const double*, const uint32_t*, size_t) {});
  uint32_t* droot; CK(cudaMalloc(&droot, (size_t)nparent * 4));
  GpuTimer t; t.start(); k_uf_roots<<<(nparent + 255) / 256, 256>>>(parent, nparent, droot); uf_ms += t.stop();
  std::vector<uint32_t> root(nparent);
  CK(cudaMemcpy(root.data(), droot, (size_t)nparent * 4, cudaMemcpyDeviceToHost));
  cudaFree(parent); cudaFree(droot);
  if (cs) { cs->join = js; cs->uf_ms = uf_ms; }
  return root;
}
