#pragma once
// Exact deduplication of identical geometries (stacked parcels: condos, multiple tax records per footprint).
// The GPU joins only distinct geometries; a result (ua, ub) expands to members(ua) x members(ub).
#include "gset.h"
#include <numeric>
#ifdef _OPENMP
#include <parallel/algorithm>
#define DD_SORT __gnu_parallel::sort
#else
#define DD_SORT std::sort
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

struct Dedup {
  HostSet u;                       // distinct geometries, in order of first occurrence
  std::vector<uint64_t> moff;      // [u.n+1]
  std::vector<uint32_t> midx;      // original indices; members of u-geometry g: midx[moff[g] .. moff[g+1])
};

static inline uint64_t geom_hash(const HostSet& s, uint64_t g) {
  uint64_t h = 1469598103934665603ull ^ (s.gro[g + 1] - s.gro[g]);
  for (uint64_t r = s.gro[g]; r < s.gro[g + 1]; r++) {
    h = (h ^ (s.rvo[r + 1] - s.rvo[r])) * 1099511628211ull;
    const uint64_t* w = reinterpret_cast<const uint64_t*>(s.xy.data() + 2 * s.rvo[r]);
    for (uint64_t i = 0; i < 2 * (s.rvo[r + 1] - s.rvo[r]); i++) { h ^= w[i]; h *= 0x100000001b3ull; h ^= h >> 29; }
  }
  return h;
}

static inline bool geom_equal(const HostSet& s, uint64_t a, uint64_t b) {
  if (s.gro[a + 1] - s.gro[a] != s.gro[b + 1] - s.gro[b]) return false;
  for (uint64_t k = 0; k < s.gro[a + 1] - s.gro[a]; k++) {
    uint64_t ra = s.gro[a] + k, rb = s.gro[b] + k;
    if (s.rvo[ra + 1] - s.rvo[ra] != s.rvo[rb + 1] - s.rvo[rb]) return false;
  }
  uint64_t va = s.rvo[s.gro[a]], vb = s.rvo[s.gro[b]], nv = s.rvo[s.gro[a + 1]] - va;
  return memcmp(s.xy.data() + 2 * va, s.xy.data() + 2 * vb, nv * 16) == 0;
}

static inline uint64_t spread32(uint64_t v) {
  v &= 0xFFFFFFFFull;
  v = (v | (v << 16)) & 0x0000FFFF0000FFFFull; v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
  v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full; v = (v | (v << 2)) & 0x3333333333333333ull;
  v = (v | (v << 1)) & 0x5555555555555555ull; return v;
}

// morton: store distinct geometries in Z-order of their bbox centres (coherent GPU traversal and memory
// access) instead of first-occurrence order.
inline Dedup dedup_set(const HostSet& in, bool morton = false) {
  const int64_t n = (int64_t)in.n;
  std::vector<uint64_t> h(n);
#pragma omp parallel for schedule(static)
  for (int64_t g = 0; g < n; g++) h[g] = geom_hash(in, g);
  std::vector<uint32_t> ord(n);
  std::iota(ord.begin(), ord.end(), 0u);
  DD_SORT(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return h[a] != h[b] ? h[a] < h[b] : a < b; });
  // rep[g] = first (lowest-index) geometry exactly equal to g
  std::vector<uint32_t> rep(n);
  for (int64_t i = 0; i < n;) {
    int64_t j = i;
    while (j < n && h[ord[j]] == h[ord[i]]) j++;
    for (int64_t k = i; k < j; k++) {        // within a hash run: verify (collisions are split off)
      uint32_t g = ord[k]; rep[g] = g;
      for (int64_t m = i; m < k; m++) if (rep[ord[m]] == ord[m] && geom_equal(in, ord[m], g)) { rep[g] = ord[m]; break; }
    }
    i = j;
  }
  Dedup d;
  std::vector<uint32_t> uid(n, UINT32_MAX);
  std::vector<uint32_t> first;              // u index -> original representative
  for (int64_t g = 0; g < n; g++) if (rep[g] == (uint32_t)g) { uid[g] = (uint32_t)first.size(); first.push_back((uint32_t)g); }
  uint64_t un = first.size();
  if (morton && un > 1) {
    std::vector<double> cx(un), cy(un);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)un; i++) {
      uint32_t g = first[i]; double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
      for (uint64_t v = in.rvo[in.gro[g]]; v < in.rvo[in.gro[g + 1]]; v++) {
        x0 = std::min(x0, in.xy[2 * v]); x1 = std::max(x1, in.xy[2 * v]); y0 = std::min(y0, in.xy[2 * v + 1]); y1 = std::max(y1, in.xy[2 * v + 1]);
      }
      cx[i] = 0.5 * (x0 + x1); cy[i] = 0.5 * (y0 + y1);
    }
    double mx0 = *std::min_element(cx.begin(), cx.end()), mx1 = *std::max_element(cx.begin(), cx.end());
    double my0 = *std::min_element(cy.begin(), cy.end()), my1 = *std::max_element(cy.begin(), cy.end());
    double sx = mx1 > mx0 ? 4294967295.0 / (mx1 - mx0) : 0, sy = my1 > my0 ? 4294967295.0 / (my1 - my0) : 0;
    std::vector<std::pair<uint64_t, uint32_t>> key(un);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)un; i++)
      key[i] = {(spread32((uint64_t)((cx[i] - mx0) * sx)) << 1) | spread32((uint64_t)((cy[i] - my0) * sy)), first[i]};
    DD_SORT(key.begin(), key.end());
    for (uint64_t i = 0; i < un; i++) { first[i] = key[i].second; uid[first[i]] = (uint32_t)i; }
  }
  std::vector<uint64_t> cnt(un + 1, 0);
  for (int64_t g = 0; g < n; g++) cnt[uid[rep[g]] + 1]++;
  d.moff.assign(un + 1, 0);
  for (uint64_t i = 0; i < un; i++) d.moff[i + 1] = d.moff[i] + cnt[i + 1];
  d.midx.resize(n);
  std::vector<uint64_t> fill(d.moff.begin(), d.moff.end() - 1);
  for (int64_t g = 0; g < n; g++) d.midx[fill[uid[rep[g]]]++] = (uint32_t)g;
  // distinct geometry set
  HostSet& u = d.u;
  u.n = un;
  // distinct geometries: offsets first, then copy in parallel
  std::vector<uint64_t> gr(un + 1, 0), gv(un + 1, 0);
  for (uint64_t i = 0; i < un; i++) {
    uint32_t g = first[i];
    gr[i + 1] = gr[i] + (in.gro[g + 1] - in.gro[g]);
    gv[i + 1] = gv[i] + (in.rvo[in.gro[g + 1]] - in.rvo[in.gro[g]]);
  }
  u.nr = gr[un]; u.nv = gv[un];
  u.gro.resize(un + 1); u.rvo.resize(u.nr + 1); u.xy.resize(2 * u.nv);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t i = 0; i < (int64_t)un; i++) {
    uint32_t g = first[i];
    u.gro[i] = gr[i];
    uint64_t r0 = in.gro[g], vbase = in.rvo[r0];
    for (uint64_t r = r0; r < in.gro[g + 1]; r++) u.rvo[gr[i] + (r - r0)] = gv[i] + (in.rvo[r] - vbase);
    memcpy(u.xy.data() + 2 * gv[i], in.xy.data() + 2 * vbase, (gv[i + 1] - gv[i]) * 16);
  }
  u.gro[un] = u.nr; u.rvo[u.nr] = u.nv;
  return d;
}
