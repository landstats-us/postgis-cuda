#pragma once
// Host-side polygon subdivision (the GPU analogue of ST_Subdivide).
//
// A geometry with more than `maxv` vertices is split recursively at the middle of the longer side of its
// bbox; every ring is clipped against the two half-planes (Sutherland-Hodgman, FP64). The pieces partition
// the polygon, so  area(A ∩ B) = Σ area(A_i ∩ B_j)  and  intersects(A, B) = OR intersects(A_i, B_j),
// provided edges created by a cut ("synthetic" edges, which are not real boundary) are ignored by the
// boundary-contact test. Ring orientation is preserved by clipping.
#include "gset.h"
#ifdef _OPENMP
#include <omp.h>
#endif

struct SubRing { std::vector<double> xy; std::vector<uint8_t> syn; };   // closed ring; syn[i]: edge i->i+1 synthetic

static void clip_ring(const SubRing& r, int axis, double m, bool keep_le, SubRing& out) {
  out.xy.clear(); out.syn.clear();
  size_t n = r.xy.size() / 2;              // includes closing vertex
  auto inside = [&](size_t i) { double v = r.xy[2 * i + axis]; return keep_le ? v <= m : v >= m; };
  auto push = [&](double x, double y, uint8_t f) {
    size_t k = out.xy.size();
    if (k >= 2 && out.xy[k - 2] == x && out.xy[k - 1] == y) { out.syn.back() = f; return; }   // drop duplicates
    out.xy.push_back(x); out.xy.push_back(y); out.syn.push_back(f);
  };
  for (size_t i = 0; i + 1 < n; i++) {
    double px = r.xy[2 * i], py = r.xy[2 * i + 1], qx = r.xy[2 * i + 2], qy = r.xy[2 * i + 3];
    bool pin = inside(i), qin = inside(i + 1);
    uint8_t f = r.syn[i];
    auto isect = [&](double& ix, double& iy) {
      if (axis == 0) { ix = m; iy = py + (m - px) * (qy - py) / (qx - px); }
      else { iy = m; ix = px + (m - py) * (qx - px) / (qy - py); }
    };
    if (pin) push(px, py, f);
    if (pin && !qin) { double ix, iy; isect(ix, iy); push(ix, iy, 1); }      // exit: next edge runs along the cut
    else if (!pin && qin) { double ix, iy; isect(ix, iy); push(ix, iy, f); }  // entry
  }
  size_t k = out.xy.size() / 2;
  if (k < 3) { out.xy.clear(); out.syn.clear(); return; }
  if (out.xy[0] != out.xy[2 * k - 2] || out.xy[1] != out.xy[2 * k - 1]) {   // close the ring
    out.xy.push_back(out.xy[0]); out.xy.push_back(out.xy[1]); out.syn.push_back(0);
  } else {
    out.syn.back() = 0;
  }
  k = out.xy.size() / 2;
  double a = 0;
  for (size_t i = 0; i + 1 < k; i++) a += out.xy[2 * i] * out.xy[2 * i + 3] - out.xy[2 * i + 2] * out.xy[2 * i + 1];
  if (k < 4 || a == 0.0) { out.xy.clear(); out.syn.clear(); }
}

static size_t piece_nverts(const std::vector<SubRing>& p) { size_t n = 0; for (auto& r : p) n += r.xy.size() / 2; return n; }

static void subdivide_rec(std::vector<SubRing>& rings, int maxv, int depth, std::vector<std::vector<SubRing>>& out) {
  size_t nv = piece_nverts(rings);
  if (nv == 0) return;
  double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
  for (auto& r : rings) for (size_t i = 0; i < r.xy.size(); i += 2) {
    x0 = std::min(x0, r.xy[i]); x1 = std::max(x1, r.xy[i]); y0 = std::min(y0, r.xy[i + 1]); y1 = std::max(y1, r.xy[i + 1]);
  }
  if ((int)nv <= maxv || depth >= 24 || (x1 - x0 <= 0 && y1 - y0 <= 0)) { out.push_back(std::move(rings)); return; }
  int axis = (x1 - x0) >= (y1 - y0) ? 0 : 1;
  double m = axis == 0 ? 0.5 * (x0 + x1) : 0.5 * (y0 + y1);
  for (int side = 0; side < 2; side++) {
    std::vector<SubRing> part; SubRing c;
    for (auto& r : rings) { clip_ring(r, axis, m, side == 0, c); if (!c.xy.empty()) part.push_back(c); }
    if (!part.empty()) subdivide_rec(part, maxv, depth + 1, out);
  }
}

// Returns a new set whose geometries are pieces; parent[piece] = index of the original geometry.
// ids are left empty (look them up through parent in the original set).
inline HostSet subdivide_set(const HostSet& in, int maxv, std::vector<uint32_t>& parent, std::vector<uint8_t>& syn) {
  const int64_t n = (int64_t)in.n;
  const int64_t CH = 1 << 15;
  int64_t nch = (n + CH - 1) / CH;
  struct Frag { std::vector<uint64_t> gro{0}, rvo{0}; std::vector<double> xy; std::vector<uint8_t> syn; std::vector<uint32_t> parent; };
  std::vector<Frag> frags(nch);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t c = 0; c < nch; c++) {
    Frag& f = frags[c];
    std::vector<std::vector<SubRing>> pieces;
    for (int64_t g = c * CH; g < std::min(n, (c + 1) * CH); g++) {
      uint64_t r0 = in.gro[g], r1 = in.gro[g + 1];
      uint64_t nv = in.rvo[r1] - in.rvo[r0];
      pieces.clear();
      if ((int64_t)nv <= maxv) {
        // fast path: copy through
        for (uint64_t r = r0; r < r1; r++) {
          uint64_t v0 = in.rvo[r], v1 = in.rvo[r + 1];
          f.xy.insert(f.xy.end(), in.xy.begin() + 2 * v0, in.xy.begin() + 2 * v1);
          f.syn.insert(f.syn.end(), v1 - v0, 0);
          f.rvo.push_back(f.xy.size() / 2);
        }
        f.gro.push_back(f.rvo.size() - 1); f.parent.push_back((uint32_t)g);
        continue;
      }
      std::vector<SubRing> rings;
      for (uint64_t r = r0; r < r1; r++) {
        uint64_t v0 = in.rvo[r], v1 = in.rvo[r + 1];
        SubRing sr; sr.xy.assign(in.xy.begin() + 2 * v0, in.xy.begin() + 2 * v1); sr.syn.assign(v1 - v0, 0);
        rings.push_back(std::move(sr));
      }
      subdivide_rec(rings, maxv, 0, pieces);
      for (auto& p : pieces) {
        for (auto& r : p) { f.xy.insert(f.xy.end(), r.xy.begin(), r.xy.end()); f.syn.insert(f.syn.end(), r.syn.begin(), r.syn.end()); f.rvo.push_back(f.xy.size() / 2); }
        f.gro.push_back(f.rvo.size() - 1); f.parent.push_back((uint32_t)g);
      }
    }
  }
  HostSet o;
  uint64_t tn = 0, tr = 0, tv = 0;
  for (auto& f : frags) { tn += f.parent.size(); tr += f.rvo.size() - 1; tv += f.xy.size() / 2; }
  o.n = tn; o.nr = tr; o.nv = tv;
  o.gro.resize(tn + 1); o.rvo.resize(tr + 1); o.xy.resize(2 * tv); syn.resize(tv); parent.resize(tn);
  std::vector<uint64_t> gbase(nch + 1, 0), rbase(nch + 1, 0), vbase(nch + 1, 0);
  for (int64_t c = 0; c < nch; c++) {
    gbase[c + 1] = gbase[c] + frags[c].parent.size(); rbase[c + 1] = rbase[c] + frags[c].rvo.size() - 1; vbase[c + 1] = vbase[c] + frags[c].xy.size() / 2;
  }
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t c = 0; c < nch; c++) {
    Frag& f = frags[c];
    for (size_t i = 0; i < f.parent.size(); i++) { o.gro[gbase[c] + i] = rbase[c] + f.gro[i]; parent[gbase[c] + i] = f.parent[i]; }
    for (size_t i = 0; i + 1 < f.rvo.size(); i++) o.rvo[rbase[c] + i] = vbase[c] + f.rvo[i];
    std::copy(f.xy.begin(), f.xy.end(), o.xy.begin() + 2 * vbase[c]);
    std::copy(f.syn.begin(), f.syn.end(), syn.begin() + vbase[c]);
    std::vector<uint64_t>().swap(f.gro); std::vector<double>().swap(f.xy);
  }
  o.gro[tn] = tr; o.rvo[tr] = tv;
  return o;
}
