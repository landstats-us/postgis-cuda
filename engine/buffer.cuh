#pragma once
// Distance joins in meters on lon/lat (EPSG:4326) layers, FP64:
//   dwithin(A, B, d)            ST_DWithin(A::geography, B::geography, d)
//   eroded intersects(ra, rb)   ST_Intersects(ST_Buffer(A::geography, -ra)::geometry, ST_Buffer(B::geography, -rb)::geometry)
// Neither builds buffer polygons.
//
// dwithin: two polygons are within d iff their real boundaries come within d of each other or one contains the
// other. Distances are measured in a local metric frame per pair (WGS84 meridional / prime-vertical radii at the
// window's latitude); boundary pairs within the frame's error of d are re-measured with Vincenty between the closest
// points. Edges are straight in lon/lat, as in the other joins (geography's edges are great-circle arcs: the
// difference is sub-millimetre for parcel-length edges).
//
// eroded intersects: X_A = ST_Buffer(A, -r) is A minus the set E_A of points within r of A's boundary, with GEOS's
// approximation of the round joins (an inscribed chord fillet with round(angle / (pi/16)) segments at every reflex
// vertex, 8 quadrant segments being GEOS's default). E_A is the union of a two-sided strip per edge, a fillet per
// reflex vertex and an (exterior) disk per convex vertex. The boundary of X_A lies on the "elements": each edge
// offset inward by r, and the fillet chords. X_A and X_B intersect iff their boundaries cross somewhere outside E_A
// and E_B, or a vertex of one eroded boundary lies inside the other eroded polygon; vertices of the eroded boundary
// are element endpoints and element crossings that lie outside E. Everything is tested in the window
// W = shrink(bbox A, ra) ∩ shrink(bbox B, rb), which contains X_A ∩ X_B. Pairs are evaluated per parent geometry
// (all pieces of a subdivided polygon at once), since the boundary that erodes a piece may belong to a neighbour.
// geography's ST_Buffer works in the UTM zone that _ST_BestSRID picks for the geometry, where r projected meters
// are r / k true meters (k = the zone's point scale factor); that is emulated per side.
// Not emulated: GEOS's input simplification (concavities shallower than r/100 are removed before offsetting, which
// moves the eroded boundary by < r/100) and its snap-rounding; results can differ from PostGIS for pairs whose eroded
// polygons are within ~r/100 of touching.
#include "engine.cuh"

constexpr double WGS84_A = 6378137.0, WGS84_F = 1.0 / 298.257223563, WGS84_E2 = WGS84_F * (2.0 - WGS84_F);
constexpr double DEG = M_PI / 180.0;

// meters per degree of longitude (kx) and of latitude (ky) at latitude lat (degrees)
__host__ __device__ __forceinline__ void m_per_deg(double lat, double& kx, double& ky) {
  double s = sin(lat * DEG), c = cos(lat * DEG);
  double w2 = 1.0 - WGS84_E2 * s * s, w = sqrt(w2);
  kx = WGS84_A / w * c * DEG;
  ky = WGS84_A * (1.0 - WGS84_E2) / (w2 * w) * DEG;
}

// Degrees that cover d meters everywhere in latitudes [y0, y1]: larger (for expanding a box) or smaller (shrinking).
__host__ __device__ __forceinline__ void deg_cover(double y0, double y1, double d, bool larger, double& dx, double& dy) {
  if (d <= 0) { dx = dy = 0; return; }
  double pad = larger ? d / 110000.0 + 1e-9 : 0.0;   // the covered points may lie up to ~d beyond the range
  double lo = (y0 - pad <= 0 && y1 + pad >= 0) ? 0.0 : fmin(fabs(y0), fabs(y1)) - pad;
  double hi = fmin(fmax(fabs(y0), fabs(y1)) + pad, 89.9);
  lo = fmax(lo, 0.0);
  double kxa, kya, kxb, kyb;
  m_per_deg(hi, kxa, kya);   // kx smallest, ky largest at the highest |lat|
  m_per_deg(lo, kxb, kyb);
  if (larger) { dx = d / kxa * (1 + 1e-6); dy = d / kyb * (1 + 1e-6); }
  else { dx = d / kxb * (1 - 1e-6); dy = d / kya * (1 - 1e-6); }
}

// Point scale factor of the UTM zone _ST_BestSRID picks for a geometry centered at (glon, glat), at (lon, lat).
// Polar areas use Lambert azimuthal equal-area (not conformal): k = 1 there.
__host__ __device__ __forceinline__ double utm_k(double glon, double glat, double lon, double lat) {
  if (glat > 70.0 || glat < -70.0) return 1.0;
  int zone = (int)floor((glon + 180.0) / 6.0);
  if (zone > 59) zone = 59;
  double lon0 = zone * 6.0 - 177.0;
  double phi = lat * DEG, c = cos(phi), t = tan(phi), T = t * t;
  double ep2 = WGS84_E2 / (1.0 - WGS84_E2), C = ep2 * c * c, A = (lon - lon0) * DEG * c, A2 = A * A;
  return 0.9996 * (1.0 + (1.0 + C) * A2 / 2.0 + (5.0 - 4.0 * T + 42.0 * C + 13.0 * C * C - 28.0 * ep2) * A2 * A2 / 24.0 +
                   (61.0 - 148.0 * T + 16.0 * T * T) * A2 * A2 * A2 / 720.0);
}

// Geodesic distance on WGS84 (Vincenty inverse), meters
__device__ double vincenty_m(double lon1, double lat1, double lon2, double lat2) {
  const double a = WGS84_A, f = WGS84_F, b = a * (1.0 - f);
  double L = (lon2 - lon1) * DEG;
  double U1 = atan((1.0 - f) * tan(lat1 * DEG)), U2 = atan((1.0 - f) * tan(lat2 * DEG));
  double sU1 = sin(U1), cU1 = cos(U1), sU2 = sin(U2), cU2 = cos(U2);
  double lam = L, sinS = 0, cosS = 1, sig = 0, cos2A = 1, cos2Sm = 0;
  for (int it = 0; it < 64; it++) {
    double sl = sin(lam), cl = cos(lam);
    double t1 = cU2 * sl, t2 = cU1 * sU2 - sU1 * cU2 * cl;
    sinS = sqrt(t1 * t1 + t2 * t2);
    if (sinS == 0) return 0.0;
    cosS = sU1 * sU2 + cU1 * cU2 * cl;
    sig = atan2(sinS, cosS);
    double sinA = cU1 * cU2 * sl / sinS;
    cos2A = 1.0 - sinA * sinA;
    cos2Sm = cos2A != 0 ? cosS - 2.0 * sU1 * sU2 / cos2A : 0.0;
    double C = f / 16.0 * cos2A * (4.0 + f * (4.0 - 3.0 * cos2A));
    double prev = lam;
    lam = L + (1.0 - C) * f * sinA * (sig + C * sinS * (cos2Sm + C * cosS * (-1.0 + 2.0 * cos2Sm * cos2Sm)));
    if (fabs(lam - prev) < 1e-14) break;
  }
  double u2 = cos2A * (a * a - b * b) / (b * b);
  double A = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)));
  double B = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)));
  double dS = B * sinS * (cos2Sm + B / 4.0 * (cosS * (-1.0 + 2.0 * cos2Sm * cos2Sm) -
                                              B / 6.0 * cos2Sm * (-3.0 + 4.0 * sinS * sinS) * (-3.0 + 4.0 * cos2Sm * cos2Sm)));
  return b * A * (sig - dS);
}

__device__ __forceinline__ double pt_seg_d2(double px, double py, double ax, double ay, double bx, double by, double& qx, double& qy) {
  double dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  double t = l2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / l2 : 0.0;
  t = fmin(1.0, fmax(0.0, t));
  qx = ax + t * dx; qy = ay + t * dy;
  return (px - qx) * (px - qx) + (py - qy) * (py - qy);
}

// squared distance between two non-crossing segments and the closest points (p on the first, q on the second)
__device__ double seg_seg_d2(double a0x, double a0y, double a1x, double a1y, double b0x, double b0y, double b1x, double b1y,
                             double& px, double& py, double& qx, double& qy) {
  double best, tx, ty;
  best = pt_seg_d2(a0x, a0y, b0x, b0y, b1x, b1y, qx, qy); px = a0x; py = a0y;
  double d = pt_seg_d2(a1x, a1y, b0x, b0y, b1x, b1y, tx, ty);
  if (d < best) { best = d; px = a1x; py = a1y; qx = tx; qy = ty; }
  d = pt_seg_d2(b0x, b0y, a0x, a0y, a1x, a1y, tx, ty);
  if (d < best) { best = d; px = tx; py = ty; qx = b0x; qy = b0y; }
  d = pt_seg_d2(b1x, b1y, a0x, a0y, a1x, a1y, tx, ty);
  if (d < best) { best = d; px = tx; py = ty; qx = b1x; qy = b1y; }
  return best;
}

// ---- geography near the threshold: PostGIS measures from the closest points of great-circle edges on the sphere
// (lwgeodetic.c edge_distance_to_point / edge_distance_to_edge), then the WGS84 geodesic between those points.
// Edges straight in lon/lat differ from great circles by up to ~L^2 tan(lat) / 8R (0.1 mm for a 100 m edge at
// 34 deg), so pairs that close to the threshold are decided this way.
struct V3 { double x, y, z; };
__device__ __forceinline__ V3 v3_ll(double lon, double lat) {   // radians
  double cl = cos(lat);
  return V3{cl * cos(lon), cl * sin(lon), sin(lat)};
}
__device__ __forceinline__ double v3_dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ V3 v3_norm(V3 a) { double l = sqrt(v3_dot(a, a)); return l > 0 ? V3{a.x / l, a.y / l, a.z / l} : a; }
__device__ __forceinline__ double v3_ang(V3 a, V3 b) {
  V3 c{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
  return atan2(sqrt(v3_dot(c, c)), v3_dot(a, b));
}
// PostGIS robust_cross_product: p x q from lon/lat (radians), accurate for nearby points
__device__ __forceinline__ V3 robust_cross(double plon, double plat, double qlon, double qlat) {
  double lon_qpp = (qlon + plon) / -2.0, lon_qmp = (qlon - plon) / 2.0;
  double smq = sin(plat - qlat), spq = sin(plat + qlat);
  double s1 = sin(lon_qpp), s2 = sin(lon_qmp), c1 = cos(lon_qpp), c2 = cos(lon_qmp);
  return V3{smq * s1 * c2 - spq * c1 * s2, smq * c1 * c2 + spq * s1 * s2, cos(plat) * cos(qlat) * sin(qlon - plon)};
}
// angle from p to the great-circle edge s-e (lon/lat radians), closest point in (clon, clat)
__device__ double arc_pt(double slon, double slat, double elon, double elat, double plon, double plat, double& clon, double& clat) {
  V3 vs = v3_ll(slon, slat), ve = v3_ll(elon, elat), vp = v3_ll(plon, plat);
  double best = 1e9;
  if (!(slon == elon && slat == elat)) {
    V3 n = v3_norm(robust_cross(slon, slat, elon, elat));
    double t = v3_dot(vp, n);
    V3 k = v3_norm(V3{vp.x - t * n.x, vp.y - t * n.y, vp.z - t * n.z});
    V3 vc = v3_norm(V3{vs.x + ve.x, vs.y + ve.y, vs.z + ve.z});   // edge_point_in_cone
    double ds = v3_dot(vs, vc), dk = v3_dot(k, vc);
    if (dk > ds || fabs(dk - ds) < 2e-16) {
      best = v3_ang(vp, k);
      clat = atan2(k.z, sqrt(k.x * k.x + k.y * k.y)); clon = atan2(k.y, k.x);
    }
  }
  double d2 = v3_ang(vp, vs), d3 = v3_ang(vp, ve);
  if (d2 < best) { best = d2; clon = slon; clat = slat; }
  if (d3 < best) { best = d3; clon = elon; clat = elat; }
  return best;
}
// geography distance (meters) of point p from edge a0-a1, or of edge a from edge b (b1 == b0 for a point): degrees in
__device__ __noinline__ double geog_dist(double a0x, double a0y, double a1x, double a1y, double b0x, double b0y, double b1x, double b1y, bool b_is_point) {
  const double R = DEG;
  double p1x, p1y, p2x, p2y, cx, cy, best;
  best = arc_pt(a0x * R, a0y * R, a1x * R, a1y * R, b0x * R, b0y * R, cx, cy); p1x = cx; p1y = cy; p2x = b0x * R; p2y = b0y * R;
  if (!b_is_point) {
    double d = arc_pt(a0x * R, a0y * R, a1x * R, a1y * R, b1x * R, b1y * R, cx, cy);
    if (d < best) { best = d; p1x = cx; p1y = cy; p2x = b1x * R; p2y = b1y * R; }
    d = arc_pt(b0x * R, b0y * R, b1x * R, b1y * R, a0x * R, a0y * R, cx, cy);
    if (d < best) { best = d; p1x = a0x * R; p1y = a0y * R; p2x = cx; p2y = cy; }
    d = arc_pt(b0x * R, b0y * R, b1x * R, b1y * R, a1x * R, a1y * R, cx, cy);
    if (d < best) { best = d; p1x = a1x * R; p1y = a1y * R; p2x = cx; p2y = cy; }
  }
  return vincenty_m(p1x / R, p1y / R, p2x / R, p2y / R);
}
// half-width (meters) of the band around d where the straight-edge frame distance may disagree with geography
// (tl = |tan(lat)| + 0.01)
__device__ __forceinline__ double geog_band(double d, double rel_frame, double La2, double Lb2, double tl) {   // squared lengths
  return d * (rel_frame + 5e-5) + (La2 + Lb2) * tl / (8.0 * 6.3e6) + 1e-9;
}

// ---------------------------------------------------------------------------------------------
// dwithin: warp per piece pair. Returns 1 / 0, or -1 when an edge list overflowed cap (*need = edges required).
// ---------------------------------------------------------------------------------------------
__device__ int dw_pair(const DSet& P, const DSet& R, uint32_t a, uint32_t b, double d, Ed* LA, Ed* LB, int cap, int lane, int* need) {
  double2 oa = P.origin[a], ob = R.origin[b], ea = P.ext[a], eb = R.ext[b];
  double sx = ob.x - oa.x, sy = ob.y - oa.y;
  double gx, gy;
  deg_cover(fmin(oa.y, ob.y), fmax(oa.y + ea.y, ob.y + eb.y), d, true, gx, gy);
  Ov ov{fmax(-gx, sx - gx), fmax(-gy, sy - gy), fmin(ea.x + gx, sx + eb.x + gx), fmin(ea.y + gy, sy + eb.y + gy)};
  if (ov.x0 > ov.x1 || ov.y0 > ov.y1) return 0;
  PV A = make_pv(P, a, oa), B = make_pv(R, b, oa);
  int na = collect_edges(A, ov, LA, lane, cap), nb = collect_edges(B, ov, LB, lane, cap);
  if (na > cap || nb > cap) { if (need) *need = max(na, nb); return -1; }
  __syncwarp();
  double phi0 = oa.y + 0.5 * (ov.y0 + ov.y1);
  double kx, ky; m_per_deg(phi0, kx, ky);
  // relative error of the frame metric over the window (scale varies with latitude), with a 2x margin
  const double tl = fabs(tan(phi0 * DEG)) + 0.01;
  double delta = 2.0 * 0.5 * (ov.y1 - ov.y0) * DEG * (tl + 0.04) + 1e-9;
  bool hit = false;
  for (int i = lane; i < na && !hit; i += 32) {
    Ed e = LA[i];
    if (e.syn) continue;
    double bx0 = fmin(e.x0, e.x1) - gx, bx1 = fmax(e.x0, e.x1) + gx, by0 = fmin(e.y0, e.y1) - gy, by1 = fmax(e.y0, e.y1) + gy;
    const double ex = (e.x1 - e.x0) * kx, ey = (e.y1 - e.y0) * ky, La2 = ex * ex + ey * ey;
    for (int j = 0; j < nb; j++) {
      Ed f = LB[j];
      if (f.syn || fmax(f.x0, f.x1) < bx0 || fmin(f.x0, f.x1) > bx1 || fmax(f.y0, f.y1) < by0 || fmin(f.y0, f.y1) > by1) continue;
      if (seg_isect(e.x0, e.y0, e.x1, e.y1, f.x0, f.y0, f.x1, f.y1)) { hit = true; break; }
      double px, py, qx, qy;
      double d2 = seg_seg_d2(e.x0 * kx, e.y0 * ky, e.x1 * kx, e.y1 * ky, f.x0 * kx, f.y0 * ky, f.x1 * kx, f.y1 * ky, px, py, qx, qy);
      double fx = (f.x1 - f.x0) * kx, fy = (f.y1 - f.y0) * ky;
      double band = geog_band(d, delta, La2, fx * fx + fy * fy, tl);
      double dlo = d - band, dhi = d + band;
      if (d2 <= dlo * dlo) { hit = true; break; }
      if (d2 <= dhi * dhi && geog_dist(oa.x + e.x0, oa.y + e.y0, oa.x + e.x1, oa.y + e.y1, oa.x + f.x0, oa.y + f.y0, oa.x + f.x1, oa.y + f.y1, false) <= d) {
        hit = true; break;
      }
    }
  }
  if (__any_sync(0xffffffffu, hit)) return 1;
  // no boundary within d: within d iff one contains the other (a real vertex of one inside the other piece)
  double cx = ov.x0 + 0.381966 * (ov.x1 - ov.x0), cy = ov.y0 + 0.414213 * (ov.y1 - ov.y0);
  bool inB_c = pip_full_warp(B, cx, cy, lane), inA_c = pip_full_warp(A, cx, cy, lane);
  bool h = false;
  for (uint32_t r = A.r0 + lane; r < A.r1 && !h; r += 32) {
    double x, y;
    if (probe_vertex(A, r, x, y) && x >= ov.x0 && x <= ov.x1 && y >= ov.y0 && y <= ov.y1 && inside_from_c(LB, nb, cx, cy, inB_c, x, y)) h = true;
  }
  for (uint32_t r = B.r0 + lane; r < B.r1 && !h; r += 32) {
    double x, y;
    if (probe_vertex(B, r, x, y) && x >= ov.x0 && x <= ov.x1 && y >= ov.y0 && y <= ov.y1 && inside_from_c(LA, na, cx, cy, inA_c, x, y)) h = true;
  }
  return __any_sync(0xffffffffu, h) ? 1 : 0;
}

constexpr int DW_WARPS = 4;
__global__ void __launch_bounds__(DW_WARPS * 32, 3) k_dwithin(const DSet P, const DSet R, const uint2* pairs, uint32_t np, double d, int same,
                                                           uint8_t* flag, uint32_t* heavy, uint32_t* nheavy, int* need) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  Ed* LA = reinterpret_cast<Ed*>(smraw) + (size_t)wib * 2 * CAP;
  Ed* LB = LA + CAP;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t p = w; p < np; p += nw) {
    if (same) {   // pieces of one geometry: within any distance of itself
      uint32_t pa = P.parent ? P.parent[pairs[p].x] : pairs[p].x, pb = R.parent ? R.parent[pairs[p].y] : pairs[p].y;
      if (pa == pb) { if (lane == 0) flag[p] = 1; continue; }
    }
    int nd = 0;
    int r = dw_pair(P, R, pairs[p].x, pairs[p].y, d, LA, LB, CAP, lane, &nd);
    if (lane == 0) {
      if (r < 0) { heavy[atomicAdd(nheavy, 1u)] = p; atomicMax(need, nd); flag[p] = 0; }
      else flag[p] = (uint8_t)r;
    }
    __syncwarp();
  }
}
// oversized pairs: per-warp edge lists in global memory
__global__ void k_dwithin_heavy(const DSet P, const DSet R, const uint2* pairs, const uint32_t* heavy, uint32_t nh, double d,
                                uint8_t* flag, Ed* scratch, int cap) {
  const int lane = threadIdx.x & 31;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  Ed* LA = scratch + (size_t)w * 2 * cap;
  Ed* LB = LA + cap;
  for (uint32_t h = w; h < nh; h += nw) {
    uint32_t p = heavy[h];
    int r = dw_pair(P, R, pairs[p].x, pairs[p].y, d, LA, LB, cap, lane, nullptr);
    if (lane == 0) flag[p] = r > 0 ? 1 : 0;
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------------------------
// eroded intersects: warp per parent pair
// ---------------------------------------------------------------------------------------------
// real boundary edge in the pair's metric frame (meters); vt describes its end vertex:
// 0 = no region (straight, or the next edge was not collected), 1 = convex (disk), m + 1 = reflex (fillet, m chords)
struct BEd { double x0, y0, x1, y1, ux, uy, L, ci, si; int succ, vt; };
struct BEl { double x0, y0, x1, y1; };   // element: piece of the raw offset curve
constexpr double FILLET_QUANTUM = M_PI / 16.0;   // GEOS: quadrant segments 8

// strictly inside (by more than eps) the region within r of the boundary; junctions between regions are closed
__device__ bool in_excl(const BEd* E, int n, double r, double x, double y, double eps) {
  if (r <= 0) return false;
  const double ri = r - eps;
  for (int k = 0; k < n; k++) {
    const BEd& e = E[k];
    if (x < fmin(e.x0, e.x1) - r || x > fmax(e.x0, e.x1) + r || y < fmin(e.y0, e.y1) - r || y > fmax(e.y0, e.y1) + r) continue;
    double dx = x - e.x0, dy = y - e.y0;
    double t = dx * e.ux + dy * e.uy, s = dy * e.ux - dx * e.uy;   // s > 0: interior (left) side
    if (fabs(s) < ri && t > -eps && t < e.L + eps) return true;   // two-sided strip along the edge
    if (e.vt == 0) continue;
    double vx = x - e.x1, vy = y - e.y1, d2 = vx * vx + vy * vy;
    if (d2 >= r * r) continue;
    const BEd& f = E[e.succ];
    if (e.vt == 1) {   // convex vertex: disk outside the interior wedge (inside it the two strips cover)
      double s2 = vy * f.ux - vx * f.uy;
      if (d2 < ri * ri && !(s > eps && s2 > eps)) return true;
      continue;
    }
    // reflex vertex: fillet polygon (v, P0 .. Pm) spanning n_k clockwise to n_s
    double nkx = -e.uy, nky = e.ux, nsx = -f.uy, nsy = f.ux;
    if (nkx * vy - nky * vx > eps) continue;    // counterclockwise of n_k
    if (nsx * vy - nsy * vx < -eps) continue;   // clockwise of n_s
    double chh = sqrt(0.5 * (1.0 + e.ci)), shh = sqrt(fmax(0.0, 0.5 * (1.0 - e.ci)));   // cos, sin of half the increment
    double ch = r * chh - eps;                   // chord distance from v
    if (d2 < ch * ch) return true;
    double mx = nkx * chh + nky * shh, my = -nkx * shh + nky * chh;   // chord normals: n_k rotated cw by (i + 1/2) inc
    bool in = true;
    for (int i = 0, m = e.vt - 1; i < m && in; i++) {
      if (mx * vx + my * vy >= ch) in = false;
      double t2 = mx * e.ci + my * e.si; my = -mx * e.si + my * e.ci; mx = t2;
    }
    if (in) return true;
  }
  return false;
}

// inside(x) = inside(c) XOR parity of edges crossing segment c -> x (edges: every real edge meeting the window)
__device__ __forceinline__ bool inside_m(const BEd* E, int n, double cx, double cy, bool in_c, double x, double y) {
  bool in = in_c;
  double sx0 = fmin(cx, x), sx1 = fmax(cx, x), sy0 = fmin(cy, y), sy1 = fmax(cy, y);
  for (int k = 0; k < n; k++) {
    const BEd& e = E[k];
    if (fmax(e.x0, e.x1) < sx0 || fmin(e.x0, e.x1) > sx1 || fmax(e.y0, e.y1) < sy0 || fmin(e.y0, e.y1) > sy1) continue;
    bool sc = left_of(e.x0, e.y0, e.x1, e.y1, cx, cy), sx = left_of(e.x0, e.y0, e.x1, e.y1, x, y);
    if (sc == sx) continue;
    bool su = orient(cx, cy, x, y, e.x0, e.y0) > 0, sv = orient(cx, cy, x, y, e.x1, e.y1) > 0;
    if (su != sv) in = !in;
  }
  return in;
}

// one side of an erosion pair: parent polygon = pieces [g0, g1) of S
struct ESide { uint32_t g0, g1; double r; double2 bmin, bmax; };

// pieces of parent p: [pf[p], pf[p + 1]) (pf: first piece per parent, from k_piece_first), or p itself if not subdivided
__device__ __forceinline__ uint2 piece_range(const uint32_t* pf, uint32_t p) { return pf ? make_uint2(pf[p], pf[p + 1]) : make_uint2(p, p + 1); }
// pf[q] = first piece whose parent is >= q, q in [0, np]; parents are sorted
__global__ void k_piece_first(const uint32_t* parent, uint32_t n, uint32_t np, uint32_t* pf) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t p = parent[i], lo = i ? parent[i - 1] + 1 : 0;
  for (uint32_t q = lo; q <= p; q++) pf[q] = i;
  if (i == n - 1) for (uint32_t q = p + 1; q <= np; q++) pf[q] = n;
}

// frame coordinates (meters) of vertex v of piece g; identical for a vertex shared by pieces of one parent
struct EFrame { double2 o; double kx, ky; };
__device__ __forceinline__ void fv(const DSet& S, double shx, double shy, uint64_t v, const EFrame& F, double& x, double& y) {
  x = (shx + S.xy[2 * (size_t)v]) * F.kx;
  y = (shy + S.xy[2 * (size_t)v + 1]) * F.ky;
}

// inside(c) for the parent: c lies in exactly one piece (simulation of simplicity on shared cut edges), so test the
// pieces whose bbox contains c, each by crossing parity over all its edges from a point left of the piece
__device__ bool pip_parent(const DSet& S, const ESide& s, const EFrame& F, double cx, double cy, int lane) {
  bool in = false;
  for (uint32_t g = s.g0; g < s.g1 && !in; g++) {
    double2 o = S.origin[g], e = S.ext[g];
    double bx0 = (o.x - F.o.x) * F.kx, by0 = (o.y - F.o.y) * F.ky, bx1 = (o.x + e.x - F.o.x) * F.kx, by1 = (o.y + e.y - F.o.y) * F.ky;
    if (cx < bx0 - 1e-6 || cx > bx1 + 1e-6 || cy < by0 - 1e-6 || cy > by1 + 1e-6) continue;
    double shx = o.x - F.o.x, shy = o.y - F.o.y, ox = bx0 - 1.0;
    bool par = false;
    for (uint32_t r = S.gro[g]; r < S.gro[g + 1]; r++) {
      uint64_t s0 = S.rvo[r], ee = S.rvo[r + 1];
      for (uint64_t j = s0 + 1 + lane; j < ee; j += 32) {
        double x0, y0, x1, y1; fv(S, shx, shy, j - 1, F, x0, y0); fv(S, shx, shy, j, F, x1, y1);
        if ((y0 > cy) == (y1 > cy) && y0 != cy && y1 != cy) continue;
        bool so = left_of(x0, y0, x1, y1, ox, cy), sc = left_of(x0, y0, x1, y1, cx, cy);
        if (so == sc) continue;
        bool su = orient(ox, cy, cx, cy, x0, y0) > 0, sv = orient(ox, cy, cx, cy, x1, y1) > 0;
        if (su != sv) par = !par;
      }
    }
    in = (__popc(__ballot_sync(0xffffffffu, par)) & 1) != 0;
  }
  return in;
}

// Collect the parent's real edges meeting box C (frame meters) into E; returns the count (may exceed cap: overflow)
__device__ int collect_side(const DSet& S, const ESide& s, const EFrame& F, double cx0, double cy0, double cx1, double cy1, BEd* E, int cap, int lane) {
  int cnt = 0;
  for (uint32_t g = s.g0; g < s.g1; g++) {
    double2 o = S.origin[g], ex = S.ext[g];
    if ((o.x + ex.x - F.o.x) * F.kx < cx0 || (o.x - F.o.x) * F.kx > cx1 || (o.y + ex.y - F.o.y) * F.ky < cy0 || (o.y - F.o.y) * F.ky > cy1) continue;
    double shx = o.x - F.o.x, shy = o.y - F.o.y;
    for (uint32_t r = S.gro[g]; r < S.gro[g + 1]; r++) {
      uint64_t s0 = S.rvo[r], e = S.rvo[r + 1];
      for (uint64_t base = s0; base + 1 < e; base += 32) {
        uint64_t i = base + lane;
        bool v = false; BEd ed;
        if (i + 1 < e && !(S.synb && ((S.synb[i >> 5] >> (i & 31)) & 1u))) {
          fv(S, shx, shy, i, F, ed.x0, ed.y0); fv(S, shx, shy, i + 1, F, ed.x1, ed.y1);
          v = !(ed.x0 == ed.x1 && ed.y0 == ed.y1) &&
              !(fmax(ed.x0, ed.x1) < cx0 || fmin(ed.x0, ed.x1) > cx1 || fmax(ed.y0, ed.y1) < cy0 || fmin(ed.y0, ed.y1) > cy1);
        }
        unsigned m = __ballot_sync(0xffffffffu, v);
        int pos = cnt + __popc(m & ((1u << lane) - 1));
        if (v && pos < cap) E[pos] = ed;
        cnt += __popc(m);
      }
    }
  }
  return cnt;
}

// Directions, successors and end-vertex regions of the collected edges
__device__ void prepare_side(BEd* E, int n, double r, int lane) {
  for (int k = lane; k < n; k += 32) {
    double dx = E[k].x1 - E[k].x0, dy = E[k].y1 - E[k].y0, L = sqrt(dx * dx + dy * dy);
    E[k].ux = dx / L; E[k].uy = dy / L; E[k].L = L;
  }
  __syncwarp();
  for (int k = lane; k < n; k += 32) {
    BEd e = E[k];
    int s = -1;
    if (k + 1 < n && E[k + 1].x0 == e.x1 && E[k + 1].y0 == e.y1) s = k + 1;
    else for (int j = 0; j < n; j++) if (j != k && E[j].x0 == e.x1 && E[j].y0 == e.y1) { s = j; break; }
    int vt = 0; double ci = 1, si = 0;
    if (s >= 0 && r > 0) {
      const BEd& f = E[s];
      double cr = e.ux * f.uy - e.uy * f.ux, dt = e.ux * f.ux + e.uy * f.uy;
      if (cr > 0) vt = 1;
      else if (cr < 0 || dt < 0) {   // right turn, or a reversal (spike): fillet (GEOS joins a reversal with a half circle)
        double th = atan2(-cr, dt);
        if (cr == 0) th = M_PI;
        int m = (int)(th / FILLET_QUANTUM + 0.5);
        if (m < 1) m = 1;
        double inc = th / m; ci = cos(inc); si = sin(inc);
        vt = m + 1;
      }
    }
    E[k].succ = s; E[k].vt = vt; E[k].ci = ci; E[k].si = si;
  }
  __syncwarp();
}

// Raw offset curve elements meeting box W (± eps): offset edges and reflex fillet chords. Returns count (may exceed cap).
__device__ int build_elements(const BEd* E, int n, double r, double wx0, double wy0, double wx1, double wy1, double eps, BEl* L, int cap, int lane) {
  int cnt = 0;
  for (int base = 0; base < n; base += 32) {
    int k = base + lane;
    bool valid = k < n;
    BEd e; if (valid) e = E[k];
    int ne = valid ? 1 + (e.vt >= 2 ? e.vt - 1 : 0) : 0;
    int mx = __reduce_max_sync(0xffffffffu, ne);
    double nkx = 0, nky = 0, wxr = 0, wyr = 0, px = 0, py = 0, nsx = 0, nsy = 0;
    if (valid) { nkx = -e.uy; nky = e.ux; wxr = nkx; wyr = nky; px = e.x1 + r * nkx; py = e.y1 + r * nky;
                 if (e.vt >= 2) { nsx = -E[e.succ].uy; nsy = E[e.succ].ux; } }
    for (int it = 0; it < mx; it++) {
      bool has = it < ne; BEl el;
      if (has) {
        if (it == 0) { el.x0 = e.x0 + r * nkx; el.y0 = e.y0 + r * nky; el.x1 = e.x1 + r * nkx; el.y1 = e.y1 + r * nky; }
        else {
          double t2 = wxr * e.ci + wyr * e.si; wyr = -wxr * e.si + wyr * e.ci; wxr = t2;
          double qx, qy;
          if (it == ne - 1) { qx = e.x1 + r * nsx; qy = e.y1 + r * nsy; } else { qx = e.x1 + r * wxr; qy = e.y1 + r * wyr; }
          el.x0 = px; el.y0 = py; el.x1 = qx; el.y1 = qy; px = qx; py = qy;
        }
        has = !(fmax(el.x0, el.x1) < wx0 - eps || fmin(el.x0, el.x1) > wx1 + eps || fmax(el.y0, el.y1) < wy0 - eps || fmin(el.y0, el.y1) > wy1 + eps);
      }
      unsigned m = __ballot_sync(0xffffffffu, has);
      int pos = cnt + __popc(m & ((1u << lane) - 1));
      if (has && pos < cap) L[pos] = el;
      cnt += __popc(m);
    }
  }
  return cnt;
}

// intersection points of two intersecting segments: 1 (proper / touching) or up to 3 (collinear overlap)
__device__ __forceinline__ int seg_points(const BEl& p, const BEl& q, double* xs, double* ys) {
  double rx = p.x1 - p.x0, ry = p.y1 - p.y0, sx = q.x1 - q.x0, sy = q.y1 - q.y0;
  double den = rx * sy - ry * sx;
  if (den != 0) {
    double t = ((q.x0 - p.x0) * sy - (q.y0 - p.y0) * sx) / den;
    t = fmin(1.0, fmax(0.0, t));
    xs[0] = p.x0 + t * rx; ys[0] = p.y0 + t * ry;
    return 1;
  }
  double l2 = rx * rx + ry * ry;
  if (l2 == 0) { xs[0] = p.x0; ys[0] = p.y0; return 1; }
  double t0 = ((q.x0 - p.x0) * rx + (q.y0 - p.y0) * ry) / l2, t1 = ((q.x1 - p.x0) * rx + (q.y1 - p.y0) * ry) / l2;
  double lo = fmax(0.0, fmin(t0, t1)), hi = fmin(1.0, fmax(t0, t1));
  if (lo > hi) { double m = 0.5 * (lo + hi); lo = hi = fmin(1.0, fmax(0.0, m)); }
  xs[0] = p.x0 + lo * rx; ys[0] = p.y0 + lo * ry;
  xs[1] = p.x0 + hi * rx; ys[1] = p.y0 + hi * ry;
  xs[2] = p.x0 + 0.5 * (lo + hi) * rx; ys[2] = p.y0 + 0.5 * (lo + hi) * ry;
  return 3;
}

struct ErScratch { BEd* EA; BEd* EB; BEl* LA; BEl* LB; int capE, capL; };

// Tests X_A ∩ X_B within the window, or within its intersection with tile (world degrees) when given.
// Returns 1 / 0, or -1 when a local list overflowed (needE / needL = sizes required; *wout = the window tested).
__device__ int er_pair(const DSet& P, const DSet& R, uint2 ga, uint2 gb, double ra_nom, double rb_nom, const ErScratch& sc,
                       int lane, int* needE, int* needL, const double4* tile = nullptr, double4* wout = nullptr) {
  ESide A, B;
  A.g0 = ga.x; A.g1 = ga.y; B.g0 = gb.x; B.g1 = gb.y;
  if (A.g0 >= A.g1 || B.g0 >= B.g1) return 0;
  auto bbox = [](const DSet& S, ESide& s) {
    s.bmin = make_double2(1e300, 1e300); s.bmax = make_double2(-1e300, -1e300);
    for (uint32_t g = s.g0; g < s.g1; g++) {
      double2 o = S.origin[g], e = S.ext[g];
      s.bmin.x = fmin(s.bmin.x, o.x); s.bmin.y = fmin(s.bmin.y, o.y);
      s.bmax.x = fmax(s.bmax.x, o.x + e.x); s.bmax.y = fmax(s.bmax.y, o.y + e.y);
    }
  };
  bbox(P, A); bbox(R, B);
  // window (world degrees): both bboxes shrunk by a lower bound of the true radius (k <= 1.001 in a UTM zone)
  double dax, day, dbx, dby;
  deg_cover(A.bmin.y, A.bmax.y, ra_nom * 0.998, false, dax, day);
  deg_cover(B.bmin.y, B.bmax.y, rb_nom * 0.998, false, dbx, dby);
  double wx0 = fmax(A.bmin.x + dax, B.bmin.x + dbx), wx1 = fmin(A.bmax.x - dax, B.bmax.x - dbx);
  double wy0 = fmax(A.bmin.y + day, B.bmin.y + dby), wy1 = fmin(A.bmax.y - day, B.bmax.y - dby);
  if (tile) { wx0 = fmax(wx0, tile->x); wy0 = fmax(wy0, tile->y); wx1 = fmin(wx1, tile->z); wy1 = fmin(wy1, tile->w); }
  if (wx0 > wx1 || wy0 > wy1) return 0;
  if (wout) *wout = make_double4(wx0, wy0, wx1, wy1);
  double clon = 0.5 * (wx0 + wx1), clat = 0.5 * (wy0 + wy1);
  A.r = ra_nom > 0 ? ra_nom / utm_k(0.5 * (A.bmin.x + A.bmax.x), 0.5 * (A.bmin.y + A.bmax.y), clon, clat) : 0.0;
  B.r = rb_nom > 0 ? rb_nom / utm_k(0.5 * (B.bmin.x + B.bmax.x), 0.5 * (B.bmin.y + B.bmax.y), clon, clat) : 0.0;
  EFrame F; F.o = P.origin[A.g0]; m_per_deg(clat, F.kx, F.ky);
  // window and collect boxes in frame meters
  double Wx0 = (wx0 - F.o.x) * F.kx, Wx1 = (wx1 - F.o.x) * F.kx, Wy0 = (wy0 - F.o.y) * F.ky, Wy1 = (wy1 - F.o.y) * F.ky;
  double ext = fmax(fmax(fabs(Wx0), fabs(Wx1)), fmax(fabs(Wy0), fabs(Wy1))) + fmax(A.r, B.r);
  const double eps = 1e-9 + 1e-12 * ext;
  // the frame metric is exact at the window's center latitude; widen the collect margin for its error
  double fe = 1.0 + 2.0 * 0.5 * (wy1 - wy0) * DEG * (fabs(tan(clat * DEG)) + 0.05);
  double ma = A.r * fe + 1e-6, mb = B.r * fe + 1e-6;
  int na = collect_side(P, A, F, Wx0 - ma, Wy0 - ma, Wx1 + ma, Wy1 + ma, sc.EA, sc.capE, lane);
  int nb = collect_side(R, B, F, Wx0 - mb, Wy0 - mb, Wx1 + mb, Wy1 + mb, sc.EB, sc.capE, lane);
  if (na > sc.capE || nb > sc.capE) { if (needE) *needE = max(na, nb); return -1; }
  __syncwarp();
  prepare_side(sc.EA, na, A.r, lane);
  prepare_side(sc.EB, nb, B.r, lane);
  int la = build_elements(sc.EA, na, A.r, Wx0, Wy0, Wx1, Wy1, eps, sc.LA, sc.capL, lane);
  int lb = build_elements(sc.EB, nb, B.r, Wx0, Wy0, Wx1, Wy1, eps, sc.LB, sc.capL, lane);
  if (la > sc.capL || lb > sc.capL) { if (needE) *needE = max(na, nb); if (needL) *needL = max(la, lb); return -1; }
  __syncwarp();
  const BEd* EA = sc.EA; const BEd* EB = sc.EB; const BEl* LA = sc.LA; const BEl* LB = sc.LB;
  auto inW = [&](double x, double y) { return x >= Wx0 - eps && x <= Wx1 + eps && y >= Wy0 - eps && y <= Wy1 + eps; };
  // (a) the eroded boundaries cross
  bool hit = false;
  for (int i = lane; i < la && !hit; i += 32) {
    BEl p = LA[i];
    double px0 = fmin(p.x0, p.x1), px1 = fmax(p.x0, p.x1), py0 = fmin(p.y0, p.y1), py1 = fmax(p.y0, p.y1);
    for (int j = 0; j < lb && !hit; j++) {
      BEl q = LB[j];
      if (fmax(q.x0, q.x1) < px0 || fmin(q.x0, q.x1) > px1 || fmax(q.y0, q.y1) < py0 || fmin(q.y0, q.y1) > py1) continue;
      if (!seg_isect(p.x0, p.y0, p.x1, p.y1, q.x0, q.y0, q.x1, q.y1)) continue;
      double xs[3], ys[3]; int k = seg_points(p, q, xs, ys);
      for (int t = 0; t < k && !hit; t++)
        if (inW(xs[t], ys[t]) && !in_excl(EA, na, A.r, xs[t], ys[t], eps) && !in_excl(EB, nb, B.r, xs[t], ys[t], eps)) hit = true;
    }
  }
  if (__any_sync(0xffffffffu, hit)) return 1;
  // (b), (c): a vertex of one eroded boundary inside the other eroded polygon
  const double cx = Wx0 + 0.381966 * (Wx1 - Wx0), cy = Wy0 + 0.414213 * (Wy1 - Wy0);
  bool inA_c = pip_parent(P, A, F, cx, cy, lane);
  bool inB_c = pip_parent(R, B, F, cx, cy, lane);
  for (int side = 0; side < 2; side++) {
    const BEd* ES = side ? EB : EA; const BEd* EO = side ? EA : EB;
    const BEl* LS = side ? LB : LA;
    int ns = side ? nb : na, no = side ? na : nb, ls = side ? lb : la;
    double rs = side ? B.r : A.r, ro = side ? A.r : B.r;
    bool inO_c = side ? inA_c : inB_c;
    auto vertex_in_other = [&](double x, double y) {
      return inW(x, y) && !in_excl(ES, ns, rs, x, y, eps) && inside_m(EO, no, cx, cy, inO_c, x, y) && !in_excl(EO, no, ro, x, y, eps);
    };
    for (int i = lane; i < ls && !hit; i += 32) {
      BEl p = LS[i];
      if (vertex_in_other(p.x0, p.y0) || vertex_in_other(p.x1, p.y1)) { hit = true; break; }
      double px0 = fmin(p.x0, p.x1), px1 = fmax(p.x0, p.x1), py0 = fmin(p.y0, p.y1), py1 = fmax(p.y0, p.y1);
      for (int j = i + 1; j < ls && !hit; j++) {
        BEl q = LS[j];
        if (fmax(q.x0, q.x1) < px0 || fmin(q.x0, q.x1) > px1 || fmax(q.y0, q.y1) < py0 || fmin(q.y0, q.y1) > py1) continue;
        if (!seg_isect(p.x0, p.y0, p.x1, p.y1, q.x0, q.y0, q.x1, q.y1)) continue;
        double xs[3], ys[3]; int k = seg_points(p, q, xs, ys);
        for (int t = 0; t < k && !hit; t++) if (vertex_in_other(xs[t], ys[t])) hit = true;
      }
    }
    if (__any_sync(0xffffffffu, hit)) return 1;
  }
  return 0;
}

constexpr int ER_WARPS = 2, ER_CAPE = 64, ER_CAPL = 128;
constexpr size_t ER_SMEM_WARP = 2 * ER_CAPE * sizeof(BEd) + 2 * ER_CAPL * sizeof(BEl);
constexpr size_t ER_SMEM = (size_t)ER_WARPS * ER_SMEM_WARP;
constexpr int ER_MAX_DEPTH = 12;
// a pair whose local lists overflowed, restricted to a tile of its window (world degrees)
struct ETile { uint32_t pair, depth; double4 box; };
struct ErCtr { uint32_t ntile, nslab; int needE, needL; };

__device__ __forceinline__ ErScratch er_smem(unsigned char* smraw, int wib) {
  unsigned char* base = smraw + (size_t)wib * ER_SMEM_WARP;
  ErScratch sc;
  sc.EA = reinterpret_cast<BEd*>(base); sc.EB = sc.EA + ER_CAPE;
  sc.LA = reinterpret_cast<BEl*>(sc.EB + ER_CAPE); sc.LB = sc.LA + ER_CAPL;
  sc.capE = ER_CAPE; sc.capL = ER_CAPL;
  return sc;
}

// Tiles that overflow are split into quadrants while they are larger than the radius, else sent to the slab path.
__device__ __forceinline__ void er_overflow(uint32_t p, uint32_t depth, double4 w, double rmax, int ne, int nl, ETile* next, ETile* slab, ErCtr* c) {
  double kx, ky; m_per_deg(0.5 * (w.y + w.w), kx, ky);
  double ext = fmax((w.z - w.x) * kx, (w.w - w.y) * ky);
  if (depth < ER_MAX_DEPTH && ext > rmax) {
    double xm = 0.5 * (w.x + w.z), ym = 0.5 * (w.y + w.w);
    uint32_t k = atomicAdd(&c->ntile, 4u);
    next[k] = ETile{p, depth + 1, make_double4(w.x, w.y, xm, ym)};
    next[k + 1] = ETile{p, depth + 1, make_double4(xm, w.y, w.z, ym)};
    next[k + 2] = ETile{p, depth + 1, make_double4(w.x, ym, xm, w.w)};
    next[k + 3] = ETile{p, depth + 1, make_double4(xm, ym, w.z, w.w)};
  } else {
    slab[atomicAdd(&c->nslab, 1u)] = ETile{p, depth, w};
    atomicMax(&c->needE, ne); atomicMax(&c->needL, nl);
  }
}

// main pass: warp per parent pair; pairs that overflow go to the heavy list (then tiles)
__global__ void __launch_bounds__(ER_WARPS * 32) k_erode_main(const DSet P, const DSet R, const uint32_t* pfP, const uint32_t* pfR, const uint2* pairs,
                                                              uint32_t np, double ra, double rb, uint8_t* flag, uint32_t* heavy, uint32_t* nheavy) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const int lane = threadIdx.x & 31;
  ErScratch sc = er_smem(smraw, threadIdx.x >> 5);
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t p = w; p < np; p += nw) {
    int r = er_pair(P, R, piece_range(pfP, pairs[p].x), piece_range(pfR, pairs[p].y), ra, rb, sc, lane, nullptr, nullptr);
    if (lane == 0) {
      flag[p] = r > 0 ? 1 : 0;
      if (r < 0) heavy[atomicAdd(nheavy, 1u)] = p;
    }
    __syncwarp();
  }
}
__global__ void k_tiles_init(const uint32_t* heavy, uint32_t n, ETile* items) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) items[i] = ETile{heavy[i], 0, make_double4(-INFINITY, -INFINITY, INFINITY, INFINITY)};
}
// tiles of overflowing pairs, shared-memory lists
__global__ void __launch_bounds__(ER_WARPS * 32) k_erode_tiles(const DSet P, const DSet R, const uint32_t* pfP, const uint32_t* pfR, const uint2* pairs, const ETile* items, uint32_t ni,
                                                               double ra, double rb, uint8_t* flag, ETile* next, ETile* slab, ErCtr* c) {
  extern __shared__ __align__(16) unsigned char smraw[];
  const int lane = threadIdx.x & 31;
  ErScratch sc = er_smem(smraw, threadIdx.x >> 5);
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  for (uint32_t i = w; i < ni; i += nw) {
    ETile t = items[i];
    if (flag[t.pair]) continue;   // another tile already found an intersection
    int ne = 0, nl = 0; double4 win;
    int r = er_pair(P, R, piece_range(pfP, pairs[t.pair].x), piece_range(pfR, pairs[t.pair].y), ra, rb, sc, lane, &ne, &nl, &t.box, &win);
    if (lane == 0) {
      if (r > 0) flag[t.pair] = 1;
      else if (r < 0) er_overflow(t.pair, t.depth, win, fmax(ra, rb), ne, nl, next, slab, c);
    }
    __syncwarp();
  }
}
// tiles that cannot be split usefully: per-warp lists in global memory; tiles that overflow again are re-queued
__global__ void k_erode_slab(const DSet P, const DSet R, const uint32_t* pfP, const uint32_t* pfR, const uint2* pairs, const ETile* items, uint32_t ni, double ra, double rb,
                             uint8_t* flag, unsigned char* scratch, int capE, int capL, ETile* again, ErCtr* c) {
  const int lane = threadIdx.x & 31;
  uint32_t w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
  unsigned char* base = scratch + (size_t)w * (2 * (size_t)capE * sizeof(BEd) + 2 * (size_t)capL * sizeof(BEl));
  ErScratch sc;
  sc.EA = reinterpret_cast<BEd*>(base); sc.EB = sc.EA + capE;
  sc.LA = reinterpret_cast<BEl*>(sc.EB + capE); sc.LB = sc.LA + capL;
  sc.capE = capE; sc.capL = capL;
  for (uint32_t i = w; i < ni; i += nw) {
    ETile t = items[i];
    if (flag[t.pair]) continue;
    int ne = 0, nl = 0;
    int r = er_pair(P, R, piece_range(pfP, pairs[t.pair].x), piece_range(pfR, pairs[t.pair].y), ra, rb, sc, lane, &ne, &nl, &t.box, nullptr);
    if (lane == 0) {
      if (r > 0) flag[t.pair] = 1;
      else if (r < 0) { again[atomicAdd(&c->nslab, 1u)] = t; atomicMax(&c->needE, ne); atomicMax(&c->needL, nl); }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------------------------
struct BufferQuery {
  enum Mode { DWITHIN, ERODE } mode = DWITHIN;
  double d = 0;          // DWITHIN: distance, meters
  double ra = 0, rb = 0; // ERODE: erosion radius of each side, meters (>= 0; 0 = not buffered)
  int same = 0;          // P and R are the same set (self-join): pairs of a geometry with itself are shortcut
  const uint8_t* need_self = nullptr;   // ERODE + same: per parent, whether its self pair is needed (nullptr: all)
};
struct BufferStats {
  double traverse_ms = 0, refine_ms = 0, compact_ms = 0, copy_ms = 0;
  double pass1_ms = 0, erode_main_ms = 0, erode_tiles_ms = 0;   // erode: intersects + area join, main erosion pass, tiles
  uint64_t candidates = 0, pairs = 0, hits = 0, heavy = 0, tiles = 0, slab_tiles = 0;
};

// query bboxes of P expanded by d meters
__global__ void k_expand_bb(const DSet P, double d, float4* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P.n) return;
  double2 o = P.origin[i], e = P.ext[i];
  double dx, dy; deg_cover(o.y, o.y + e.y, d, true, dx, dy);
  double x0 = o.x - dx, y0 = o.y - dy, x1 = o.x + e.x + dx, y1 = o.y + e.y + dy;
  float fx0 = (float)x0, fy0 = (float)y0, fx1 = (float)x1, fy1 = (float)y1;
  if ((double)fx0 > x0) fx0 = nextafterf(fx0, -INFINITY);
  if ((double)fy0 > y0) fy0 = nextafterf(fy0, -INFINITY);
  if ((double)fx1 < x1) fx1 = nextafterf(fx1, INFINITY);
  if ((double)fy1 < y1) fy1 = nextafterf(fy1, INFINITY);
  out[i] = make_float4(fx0, fy0, fx1, fy1);
}
__global__ void k_flag_is(const uint8_t* f, uint32_t n, uint8_t* o) { uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = f[i] != 0; }

// erosion pass 2 input: intersecting parent pairs whose overlap area can hold a disk of radius rmin (meters;
// 0: any overlap; < 0: any intersecting pair), self pairs only where needed
__global__ void k_er_filter(const uint2* pr, const double* area, uint32_t n, const DSet P, const uint32_t* pfP, double rmin,
                            int same, const uint8_t* need_self, uint8_t* keep) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint2 p = pr[i];
  bool k = true;
  if (same && p.x == p.y && need_self) k = need_self[p.x] != 0;
  if (k && rmin >= 0) {
    uint32_t g = pfP ? pfP[p.x] : p.x;
    double y0 = P.origin[g].y, y1 = y0 + P.ext[g].y;
    double lat = (y0 <= 0 && y1 >= 0) ? 0.0 : (fabs(y0) < fabs(y1) ? y0 : y1);
    double kx, ky; m_per_deg(lat, kx, ky);
    double a_m2 = area[i] * kx * ky * 1.001;   // upper bound over the pair's latitudes
    // GEOS's eroded polygon reaches up to ~r/100 (input simplification) + 0.5% (chord fillets) closer to the boundary
    double r = 0.98 * rmin;
    k = rmin > 0 ? a_m2 >= M_PI * r * r : area[i] > 0;
  }
  keep[i] = k;
}

// candidate piece pairs whose eroded polygons cannot meet: both pieces are whole (single-piece) geometries and their
// bboxes shrunk by (a lower bound of) the radii do not overlap
__global__ void k_er_prefilter(const DSet P, const DSet R, const uint2* pairs, uint32_t n, double ra, double rb, uint8_t* keep) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint2 p = pairs[i];
  auto whole = [](const DSet& S, uint32_t g) {
    if (!S.parent) return true;
    uint32_t q = S.parent[g];
    return (g == 0 || S.parent[g - 1] != q) && (g + 1 >= S.n || S.parent[g + 1] != q);
  };
  bool k = true;
  if (whole(P, p.x) && whole(R, p.y)) {
    double2 oa = P.origin[p.x], ea = P.ext[p.x], ob = R.origin[p.y], eb = R.ext[p.y];
    double dax, day, dbx, dby;
    deg_cover(oa.y, oa.y + ea.y, ra * 0.998, false, dax, day);
    deg_cover(ob.y, ob.y + eb.y, rb * 0.998, false, dbx, dby);
    double wx0 = fmax(oa.x + dax, ob.x + dbx), wx1 = fmin(oa.x + ea.x - dax, ob.x + eb.x - dbx);
    double wy0 = fmax(oa.y + day, ob.y + dby), wy1 = fmin(oa.y + ea.y - day, ob.y + eb.y - dby);
    k = wx0 <= wx1 && wy0 <= wy1;
  }
  keep[i] = k;
}

// erode: X_A ∩ X_B over intersecting pairs. Pass 1 is the exact intersects + area join; pass 2 tests erosion.
template <class F>
BufferStats erode_join_stream(const DSet& P, const DSet& R, const Bvh& bvh, const BufferQuery& q, JoinOpts o, F&& cb) {
  BufferStats st;
  if (P.n == 0 || R.n == 0) return st;
  DevBuf b_pfP, b_pfR, b_keep, b_ep, b_flag, b_heavy, b_again, b_again2, b_tiles, b_tiles2, b_slab, b_ctr, b_ctr2, b_tmp, b_nsel, b_out, b_scratch;
  PinnedBuf h_pairs;
  static const bool dbg = getenv("BUF_DEBUG") != nullptr;
  auto make_pf = [](const DSet& S, DevBuf& b) -> const uint32_t* {
    if (!S.parent) return nullptr;
    uint32_t np = S.h_parent[S.n - 1] + 1;
    uint32_t* pf = (uint32_t*)b.get(((size_t)np + 1) * 4);
    k_piece_first<<<(S.n + 255) / 256, 256>>>(S.parent, S.n, np, pf);
    CK(cudaGetLastError());
    return pf;
  };
  const uint32_t* pfP = make_pf(P, b_pfP);
  const uint32_t* pfR = R.parent ? make_pf(R, b_pfR) : nullptr;
  CK(cudaFuncSetAttribute(k_erode_main, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ER_SMEM));
  CK(cudaFuncSetAttribute(k_erode_tiles, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ER_SMEM));
  const double rmin = (q.ra > 0 && q.rb > 0) ? std::min(q.ra, q.rb) : (q.ra > 0 || q.rb > 0 ? 0.0 : -1.0);
  GpuTimer tm;
  double er_ms = 0, er_compact_ms = 0;
  auto heavy_warps = [](uint32_t nh, size_t per) {
    size_t w = std::max<size_t>(1, std::min<size_t>(nh, (size_t)(1ull << 30) / per));
    return (uint32_t)((w + 3) / 4 * 4);
  };
  RefineParams rp; rp.want_area = 1;
  JoinOpts o1 = o; o1.device_only = true;
  DevBuf b_pk, b_pp, b_pn, b_ptmp;
  if (q.ra > 0 || q.rb > 0) o1.prefilter = [&](uint2* pairs, uint32_t n) -> uint32_t {
    uint8_t* keep = (uint8_t*)b_pk.get((size_t)n + 1);
    k_er_prefilter<<<(n + 255) / 256, 256>>>(P, R, pairs, n, q.ra, q.rb, keep);
    uint2* tmpp = (uint2*)b_pp.get((size_t)n * 8 + 8);
    int* nsel = (int*)b_pn.get(8);
    size_t tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, pairs, keep, tmpp, nsel, (int)n);
    cub::DeviceSelect::Flagged(b_ptmp.get(tmp), tmp, pairs, keep, tmpp, nsel, (int)n);
    int k = 0; CK(cudaMemcpy(&k, nsel, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pairs, tmpp, (size_t)k * 8, cudaMemcpyDeviceToDevice));
    return (uint32_t)k;
  };
  o1.on_device = [&](const uint2* dp, const double* da, uint32_t n) {
    tm.start();
    int* nsel = (int*)b_nsel.get(8);
    uint8_t* keep = (uint8_t*)b_keep.get((size_t)n + 1);
    k_er_filter<<<(n + 255) / 256, 256>>>(dp, da, n, P, pfP, rmin, q.same, q.need_self, keep);
    uint2* ep = (uint2*)b_ep.get((size_t)n * 8 + 8);
    size_t tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, dp, keep, ep, nsel, (int)n);
    cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, dp, keep, ep, nsel, (int)n);
    int ne = 0; CK(cudaMemcpy(&ne, nsel, 4, cudaMemcpyDeviceToHost));
    er_compact_ms += tm.stop();
    st.pairs += ne;
    if (!ne) return;
    tm.start();
    uint8_t* flag = (uint8_t*)b_flag.get((size_t)ne + 1);
    uint32_t* heavy = (uint32_t*)b_heavy.get((size_t)ne * 4 + 4);
    uint32_t* ctr = (uint32_t*)b_ctr.get(16);
    CK(cudaMemset(ctr, 0, 16));
    int G = (int)std::min<uint64_t>(((uint64_t)ne * 32 + 63) / 64, 132 * 32);
    k_erode_main<<<G, ER_WARPS * 32, ER_SMEM>>>(P, R, pfP, pfR, ep, ne, q.ra, q.rb, flag, heavy, ctr);
    CK(cudaGetLastError());
    uint32_t nh = 0; CK(cudaMemcpy(&nh, ctr, 4, cudaMemcpyDeviceToHost));
    { double ms = tm.stop(); st.erode_main_ms += ms; er_ms += ms; tm.start();
      if (dbg) fprintf(stderr, "erode: pass-1 pairs=%u -> tested %d, heavy %u  %.1f ms\n", n, ne, nh, ms); }
    st.heavy += nh;
    if (nh) {
      // overflowing pairs: split their windows into tiles until each fits the shared-memory lists
      ErCtr* ec = (ErCtr*)b_ctr2.get(sizeof(ErCtr));
      uint32_t ni = nh;
      ETile* items = (ETile*)b_tiles.get((size_t)ni * sizeof(ETile));
      k_tiles_init<<<(ni + 255) / 256, 256>>>(heavy, ni, items);
      for (uint32_t round = 0; ni; round++) {
        ETile* next = (ETile*)b_tiles2.get((size_t)ni * 4 * sizeof(ETile));
        ETile* slab = (ETile*)b_slab.get((size_t)ni * sizeof(ETile));
        CK(cudaMemset(ec, 0, sizeof(ErCtr)));
        int GT = (int)std::min<uint64_t>(((uint64_t)ni * 32 + 63) / 64, 132 * 32);
        k_erode_tiles<<<GT, ER_WARPS * 32, ER_SMEM>>>(P, R, pfP, pfR, ep, items, ni, q.ra, q.rb, flag, next, slab, ec);
        CK(cudaGetLastError());
        ErCtr hcn; CK(cudaMemcpy(&hcn, ec, sizeof(ErCtr), cudaMemcpyDeviceToHost));
        st.tiles += ni; st.slab_tiles += hcn.nslab;
        if (dbg) fprintf(stderr, "  tiles round %u: items=%u -> split %u, slab %u\n", round, ni, hcn.ntile, hcn.nslab);
        // tiles that cannot be split usefully: global-memory lists, re-run with what they reported needing
        for (uint32_t ns = hcn.nslab, sround = 0; ns; sround++) {
          if (sround >= 4) { fprintf(stderr, "erode: %u tiles still overflow after %u rounds\n", ns, sround); exit(1); }
          int capE = std::max<int>(hcn.needE, ER_CAPE), capL = std::max<int>(std::max<int>(hcn.needL, 17 * capE / 4), ER_CAPL);
          size_t per = 2 * (size_t)capE * sizeof(BEd) + 2 * (size_t)capL * sizeof(BEl);
          uint32_t nwarps = heavy_warps(ns, per);
          unsigned char* scr = (unsigned char*)b_scratch.get(per * nwarps);
          ETile* again = (ETile*)(sround % 2 ? b_again2 : b_again).get((size_t)ns * sizeof(ETile));
          CK(cudaMemset(ec, 0, sizeof(ErCtr)));
          k_erode_slab<<<nwarps / 4, 128>>>(P, R, pfP, pfR, ep, slab, ns, q.ra, q.rb, flag, scr, capE, capL, again, ec);
          CK(cudaGetLastError());
          ErCtr h2; CK(cudaMemcpy(&h2, ec, sizeof(ErCtr), cudaMemcpyDeviceToHost));
          if (dbg) fprintf(stderr, "  slab: tiles=%u capE=%d capL=%d warps=%u again=%u\n", ns, capE, capL, nwarps, h2.nslab);
          slab = again; ns = h2.nslab; hcn.needE = h2.needE; hcn.needL = h2.needL;
        }
        std::swap(b_tiles.p, b_tiles2.p); std::swap(b_tiles.cap, b_tiles2.cap);
        items = (ETile*)b_tiles.p;
        ni = hcn.ntile;
      }
    }
    { double ms = tm.stop(); er_ms += ms; st.erode_tiles_ms += ms; }
    tm.start();
    uint8_t* sel = (uint8_t*)b_keep.get((size_t)ne + 1);
    k_flag_is<<<(ne + 255) / 256, 256>>>(flag, ne, sel);
    uint2* out = (uint2*)b_out.get((size_t)ne * 8 + 8);
    tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, ep, sel, out, nsel, ne);
    cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, ep, sel, out, nsel, ne);
    int nhit = 0; CK(cudaMemcpy(&nhit, nsel, 4, cudaMemcpyDeviceToHost));
    er_compact_ms += tm.stop();
    st.hits += nhit;
    if (!nhit) return;
    if (o.on_device) o.on_device(out, nullptr, (uint32_t)nhit);
    if (o.device_only) return;
    tm.start();
    uint2* hp = (uint2*)h_pairs.get((size_t)nhit * sizeof(uint2) + 8);
    CK(cudaMemcpy(hp, out, (size_t)nhit * sizeof(uint2), cudaMemcpyDeviceToHost));
    st.copy_ms += tm.stop();
    cb(hp, (size_t)nhit);
  };
  JoinStats js = join_stream(P, R, bvh, rp, o1, [](const uint2*, const double*, const uint32_t*, size_t) {});
  st.traverse_ms = js.traverse_ms; st.candidates = js.candidates;
  st.refine_ms = js.refine_ms + er_ms; st.compact_ms = js.compact_ms + er_compact_ms; st.pass1_ms = js.refine_ms;
  if (dbg) fprintf(stderr, "erode: pass 1 (intersects + area) refine %.1f ms, pass 2 refine %.1f ms\n", js.refine_ms, er_ms);
  return st;
}

// dwithin: warp per candidate piece pair (query boxes expanded by d), then piece pairs -> unique parent pairs
template <class F>
BufferStats dwithin_join_stream(const DSet& P, const DSet& R, const Bvh& bvh, const BufferQuery& q, JoinOpts o, F&& cb) {
  BufferStats st;
  if (P.n == 0 || R.n == 0) return st;
  DevBuf b_qbb, b_counts, b_offs, b_pairs, b_tmp, b_flag, b_heavy, b_sel, b_nsel, b_keys, b_keys2, b_vals, b_scratch, b_ctr;
  const bool agg = P.parent || R.parent;
  const uint32_t* hp = P.h_parent;
  std::vector<uint32_t> hcounts;
  PinnedBuf h_pairs;
  GpuTimer tm;
  const float4* qbb = P.bb;
  if (q.d > 0) {
    float4* e = (float4*)b_qbb.get((size_t)P.n * sizeof(float4));
    k_expand_bb<<<(P.n + 255) / 256, 256>>>(P, q.d, e);
    CK(cudaGetLastError());
    qbb = e;
  }
  const size_t dw_smem = (size_t)DW_WARPS * 2 * CAP * sizeof(Ed);
  CK(cudaFuncSetAttribute(k_dwithin, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dw_smem));
  uint32_t* ctr = (uint32_t*)b_ctr.get(16);
  for (uint32_t q0 = 0, qn; q0 < P.n; q0 = qn) {
    qn = std::min<uint32_t>(q0 + o.batch_q, P.n);
    if (hp) while (qn < P.n && hp[qn] == hp[qn - 1]) qn++;
    if (o.cancel && o.cancel->load()) break;
    uint32_t nq = qn - q0;
    uint32_t* counts = (uint32_t*)b_counts.get((size_t)nq * 4);
    uint32_t* offs = (uint32_t*)b_offs.get((size_t)nq * 4);
    tm.start();
    k_traverse<<<(nq + 127) / 128, 128>>>(qbb, q0, nq, bvh, counts, nullptr, nullptr, 0, o.upper ? 1 : 0);
    hcounts.resize(nq);
    CK(cudaMemcpy(hcounts.data(), counts, (size_t)nq * 4, cudaMemcpyDeviceToHost));
    st.traverse_ms += tm.stop();
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
      k_traverse<<<(ns + 127) / 128, 128>>>(qbb, q0 + s0, ns, bvh, counts + s0, offs, pairs, 1, o.upper ? 1 : 0);
      st.traverse_ms += tm.stop();
      st.candidates += np;
      s0 = s1;
      if (!np) continue;
      int* nsel = (int*)b_nsel.get(8);
      uint32_t n = (uint32_t)np;
      st.pairs += n;
      uint8_t* flag = (uint8_t*)b_flag.get((size_t)n + 1);
      uint32_t* heavy = (uint32_t*)b_heavy.get((size_t)n * 4 + 4);
      CK(cudaMemset(ctr, 0, 16));
      tm.start();
      int G = (int)std::min<uint64_t>(((uint64_t)n * 32 + 127) / 128, 132 * 16);
      k_dwithin<<<G, DW_WARPS * 32, dw_smem>>>(P, R, pairs, n, q.d, q.same, flag, heavy, ctr, (int*)ctr + 1);
      CK(cudaGetLastError());
      uint32_t hc[2]; CK(cudaMemcpy(hc, ctr, 8, cudaMemcpyDeviceToHost));
      st.heavy += hc[0];
      if (hc[0]) {   // oversized pairs: per-warp edge lists in global memory (1 GiB budget)
        int cap = (int)hc[1];
        size_t per = 2 * (size_t)cap * sizeof(Ed);
        size_t w = std::max<size_t>(1, std::min<size_t>(hc[0], (size_t)(1ull << 30) / per));
        uint32_t nwarps = (uint32_t)((w + 3) / 4 * 4);
        Ed* scr = (Ed*)b_scratch.get(per * nwarps);
        k_dwithin_heavy<<<nwarps / 4, 128>>>(P, R, pairs, heavy, hc[0], q.d, flag, scr, cap);
        CK(cudaGetLastError());
      }
      st.refine_ms += tm.stop();
      tm.start();
      uint8_t* keep = (uint8_t*)b_sel.get((size_t)n + 1);
      k_flag_is<<<(n + 255) / 256, 256>>>(flag, n, keep);
      uint2* opairs = (uint2*)b_vals.get((size_t)n * 8 + 8);
      tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, pairs, keep, opairs, nsel, (int)n);
      cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, pairs, keep, opairs, nsel, (int)n);
      int nh = 0; CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
      if (agg && nh) {   // piece pairs -> parent pairs (unique)
        unsigned long long* k1 = (unsigned long long*)b_keys.get((size_t)nh * 8);
        unsigned long long* k2 = (unsigned long long*)b_keys2.get((size_t)nh * 8);
        k_parent_keys<<<(nh + 255) / 256, 256>>>(opairs, nh, P.parent, R.parent, k1);
        tmp = 0; cub::DeviceRadixSort::SortKeys(nullptr, tmp, k1, k2, nh);
        cub::DeviceRadixSort::SortKeys(b_tmp.get(tmp), tmp, k1, k2, nh);
        tmp = 0; cub::DeviceSelect::Unique(nullptr, tmp, k2, k1, nsel, nh);
        cub::DeviceSelect::Unique(b_tmp.get(tmp), tmp, k2, k1, nsel, nh);
        CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
        k_unpack_keys<<<(nh + 255) / 256, 256>>>(k1, nh, opairs);
      }
      st.compact_ms += tm.stop();
      if (o.on_device && nh) o.on_device(opairs, nullptr, (uint32_t)nh);
      if (o.device_only) { st.hits += nh; continue; }
      tm.start();
      uint2* hpairs = (uint2*)h_pairs.get((size_t)nh * sizeof(uint2) + 8);
      CK(cudaMemcpy(hpairs, opairs, (size_t)nh * sizeof(uint2), cudaMemcpyDeviceToHost));
      st.copy_ms += tm.stop();
      st.hits += nh;
      if (nh) cb(hpairs, (size_t)nh);
    }
  }
  return st;
}

// cb(pairs (parent of P, parent of R), n) for every finished sub-batch
template <class F>
BufferStats buffer_join_stream(const DSet& P, const DSet& R, const Bvh& bvh, const BufferQuery& q, JoinOpts o, F&& cb) {
  if (q.mode == BufferQuery::ERODE) return erode_join_stream(P, R, bvh, q, o, std::forward<F>(cb));
  return dwithin_join_stream(P, R, bvh, q, o, std::forward<F>(cb));
}

// ---------------------------------------------------------------------------------------------
// Points x polygons: dwithin(point, polygon, d) and point in eroded polygon (ST_Intersects(ST_Buffer(poly::geography, -r), pt))
// ---------------------------------------------------------------------------------------------
// query boxes of points expanded by d meters
__global__ void k_expand_pts(const PSetD pts, double d, float4* out) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pts.n) return;
  double2 p = pts.xy[i];
  double dx, dy; deg_cover(p.y, p.y, d, true, dx, dy);
  double x0 = p.x - dx, y0 = p.y - dy, x1 = p.x + dx, y1 = p.y + dy;
  float fx0 = (float)x0, fy0 = (float)y0, fx1 = (float)x1, fy1 = (float)y1;
  if ((double)fx0 > x0) fx0 = nextafterf(fx0, -INFINITY);
  if ((double)fy0 > y0) fy0 = nextafterf(fy0, -INFINITY);
  if ((double)fx1 < x1) fx1 = nextafterf(fx1, INFINITY);
  if ((double)fy1 < y1) fy1 = nextafterf(fy1, INFINITY);
  out[i] = make_float4(fx0, fy0, fx1, fy1);
}

// thread per (point, piece): inside the piece, or within d of its real boundary
__global__ void k_pt_dwithin(const PSetD pts, const DSet R, const uint2* pairs, uint32_t np, double d, uint8_t* hit) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  uint2 pr = pairs[i];
  double2 pt = pts.xy[pr.x], o = R.origin[pr.y];
  PV B = make_pv(R, pr.y, o);
  double px = pt.x - o.x, py = pt.y - o.y;
  double kx, ky; m_per_deg(pt.y, kx, ky);
  double gy = d / ky;
  const double tl = fabs(tan(pt.y * DEG)) + 0.01;
  double delta = 2.0 * gy * DEG * (tl + 0.04) + 1e-9;
  double ox = B.shx - 1.0;
  bool par = false, h = false;
  for (uint32_t r = B.r0; r < B.r1 && !h; r++) {
    uint64_t s0 = B.rvo[r], e = B.rvo[r + 1];
    double x0, y0; B.v(s0, x0, y0);
    for (uint64_t j = s0 + 1; j < e && !h; j++) {
      double x1, y1; B.v(j, x1, y1);
      if (!B.syn(j - 1)) {
        double qx, qy;
        double d2 = pt_seg_d2(px * kx, py * ky, x0 * kx, y0 * ky, x1 * kx, y1 * ky, qx, qy);
        double ex = (x1 - x0) * kx, ey = (y1 - y0) * ky;
        double band = geog_band(d, delta, ex * ex + ey * ey, 0.0, tl), dlo = d - band, dhi = d + band;
        if (d2 <= dlo * dlo || (d2 <= dhi * dhi && geog_dist(o.x + x0, o.y + y0, o.x + x1, o.y + y1, pt.x, pt.y, pt.x, pt.y, true) <= d)) h = true;
      }
      if (!((y0 > py) == (y1 > py) && y0 != py && y1 != py)) {
        bool so = left_of(x0, y0, x1, y1, ox, py), sp = left_of(x0, y0, x1, y1, px, py);
        if (so != sp) {
          bool su = orient(ox, py, px, py, x0, y0) > 0, sv = orient(ox, py, px, py, x1, y1) > 0;
          if (su != sv) par = !par;
        }
      }
      x0 = x1; y0 = y1;
    }
  }
  hit[i] = (h || par) ? 1 : 0;
}

// Is w (meters, relative to vertex v) strictly inside the region of vertex v between edges with unit directions
// (ex, ey) -> (fx, fy)? Convex: the disk outside the interior wedge; reflex: the GEOS fillet polygon.
__device__ bool vertex_excl(double ex, double ey, double fx, double fy, double r, double wx, double wy, double eps) {
  double d2 = wx * wx + wy * wy;
  if (d2 >= r * r) return false;
  double cr = ex * fy - ey * fx, dt = ex * fx + ey * fy;
  if (cr > 0) {
    double ri = r - eps, s1 = wy * ex - wx * ey, s2 = wy * fx - wx * fy;
    return d2 < ri * ri && !(s1 > eps && s2 > eps);
  }
  if (cr == 0 && dt >= 0) return false;   // straight
  double th = cr == 0 ? M_PI : atan2(-cr, dt);
  int m = (int)(th / FILLET_QUANTUM + 0.5);
  if (m < 1) m = 1;
  double inc = th / m, ci = cos(inc), si = sin(inc);
  double nkx = -ey, nky = ex, nsx = -fy, nsy = fx;
  if (nkx * wy - nky * wx > eps || nsx * wy - nsy * wx < -eps) return false;
  double chh = sqrt(0.5 * (1.0 + ci)), shh = sqrt(fmax(0.0, 0.5 * (1.0 - ci)));
  double ch = r * chh - eps;
  if (d2 < ch * ch) return true;
  double mx = nkx * chh + nky * shh, my = -nkx * shh + nky * chh;
  for (int i = 0; i < m; i++) {
    if (mx * wx + my * wy >= ch) return false;
    double t2 = mx * ci + my * si; my = -mx * si + my * ci; mx = t2;
  }
  return true;
}

// thread per (point, piece) for every piece within r of the point: bit 1 = inside the piece, bit 2 = within r of the
// piece's real boundary (GEOS fillets at reflex vertices). OR-ed per (point, parent): in the eroded polygon iff == 1.
__global__ void k_pt_erode(const PSetD pts, const DSet R, const uint2* pairs, uint32_t np, double r_nom, uint8_t* bits) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= np) return;
  uint2 pr = pairs[i];
  uint32_t g = pr.y;
  double2 pt = pts.xy[pr.x], o = R.origin[g], ext = R.ext[g];
  double r = r_nom / utm_k(o.x + 0.5 * ext.x, o.y + 0.5 * ext.y, pt.x, pt.y);
  double kx, ky; m_per_deg(pt.y, kx, ky);
  const double eps = 1e-9;
  // frame: meters relative to the point
  double shx = o.x - pt.x, shy = o.y - pt.y, ox = shx * kx - 1.0;
  auto syn = [&](uint64_t j) { return R.synb && ((R.synb[j >> 5] >> (j & 31)) & 1u); };
  auto vx = [&](uint64_t j, double& x, double& y) { x = (shx + R.xy[2 * (size_t)j]) * kx; y = (shy + R.xy[2 * (size_t)j + 1]) * ky; };
  bool par = false, ex = false;
  for (uint32_t rr = R.gro[g]; rr < R.gro[g + 1] && !ex; rr++) {
    uint64_t s0 = R.rvo[rr], e = R.rvo[rr + 1];
    if (e - s0 < 2) continue;
    // previous real edge direction (region of the vertex that starts the current edge); the ring closes
    bool pv = false; double pux = 0, puy = 0;
    { uint64_t j = e - 2; if (!syn(j)) { double a, b, c, dd; vx(j, a, b); vx(j + 1, c, dd); double L = hypot(c - a, dd - b);
        if (L > 0) { pv = true; pux = (c - a) / L; puy = (dd - b) / L; } } }
    double x0, y0; vx(s0, x0, y0);
    for (uint64_t j = s0; j + 1 < e && !ex; j++) {
      double x1, y1; vx(j + 1, x1, y1);
      // piece parity over all its edges: segment from (ox, 0) to the point (0, 0)
      if (!((y0 > 0) == (y1 > 0) && y0 != 0 && y1 != 0)) {
        bool so = left_of(x0, y0, x1, y1, ox, 0.0), sp = left_of(x0, y0, x1, y1, 0.0, 0.0);
        if (so != sp) {
          bool su = orient(ox, 0.0, 0.0, 0.0, x0, y0) > 0, sv = orient(ox, 0.0, 0.0, 0.0, x1, y1) > 0;
          if (su != sv) par = !par;
        }
      }
      double L = hypot(x1 - x0, y1 - y0);
      if (syn(j) || L == 0) { if (syn(j)) pv = false; x0 = x1; y0 = y1; continue; }
      double ux = (x1 - x0) / L, uy = (y1 - y0) / L;
      if (r > 0) {
        double t = -x0 * ux - y0 * uy, s = -y0 * ux + x0 * uy;   // the point (0,0) relative to the edge
        if (fabs(s) < r - eps && t > -eps && t < L + eps) ex = true;
        else if (pv && vertex_excl(pux, puy, ux, uy, r, -x0, -y0, eps)) ex = true;
      }
      pv = true; pux = ux; puy = uy;
      x0 = x1; y0 = y1;
    }
  }
  bits[i] = ex ? 2 : (par ? 1 : 0);   // the scan stops early only once the point is known to be excluded
}
__global__ void k_is_one_u8(const uint8_t* f, uint32_t n, uint8_t* o) { uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = f[i] == 1; }
struct OrU8 { __host__ __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return a | b; } };

// cb(pairs (point index, parent polygon), n) for every finished batch
template <class F>
BufferStats points_buffer_join_stream(const PSetD& pts, const DSet& R, const Bvh& bvh, const BufferQuery& q, JoinOpts o, F&& cb) {
  BufferStats st;
  if (pts.n == 0 || R.n == 0) return st;
  const bool er = q.mode == BufferQuery::ERODE;
  DevBuf b_qbb, b_counts, b_offs, b_pairs, b_hit, b_tmp, b_opairs, b_nsel, b_keys, b_keys2, b_sel;
  std::vector<uint32_t> hcounts; PinnedBuf h_pairs;
  GpuTimer tm;
  // query boxes widened by d (dwithin) or by r (erode: every piece whose boundary may be within r of the point)
  const float4* qbb = pts.bb;
  const double grow = er ? q.rb * 1.002 : q.d;
  if (grow > 0) {
    float4* e = (float4*)b_qbb.get((size_t)pts.n * sizeof(float4));
    k_expand_pts<<<(pts.n + 255) / 256, 256>>>(pts, grow, e);
    CK(cudaGetLastError());
    qbb = e;
  }
  for (uint32_t q0 = 0; q0 < pts.n; q0 += o.batch_q) {
    uint32_t nq = std::min<uint32_t>(o.batch_q, pts.n - q0);
    if (o.cancel && o.cancel->load()) break;
    uint32_t* counts = (uint32_t*)b_counts.get((size_t)nq * 4);
    uint32_t* offs = (uint32_t*)b_offs.get((size_t)nq * 4);
    tm.start();
    k_traverse<<<(nq + 127) / 128, 128>>>(qbb, q0, nq, bvh, counts, nullptr, nullptr, 0);
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
      k_traverse<<<(ns + 127) / 128, 128>>>(qbb, q0 + s0, ns, bvh, counts + s0, offs, pairs, 1);
      st.traverse_ms += tm.stop(); st.candidates += np;
      s0 = s1;
      if (!np) continue;
      int* nsel = (int*)b_nsel.get(8);
      uint32_t n = (uint32_t)np;
      st.pairs += n;
      uint8_t* hit = (uint8_t*)b_hit.get((size_t)n + 1);
      tm.start();
      if (er) k_pt_erode<<<(n + 127) / 128, 128>>>(pts, R, pairs, n, q.rb, hit);
      else k_pt_dwithin<<<(n + 127) / 128, 128>>>(pts, R, pairs, n, q.d, hit);
      CK(cudaGetLastError());
      st.refine_ms += tm.stop();
      tm.start();
      uint2* opairs = (uint2*)b_opairs.get((size_t)n * 8 + 8);
      int nh = 0;
      if (er && R.parent) {   // OR the (point, piece) bits per (point, parent): inside some piece, near no boundary
        unsigned long long* k1 = (unsigned long long*)b_keys.get((size_t)n * 8);
        unsigned long long* k2 = (unsigned long long*)b_keys2.get((size_t)n * 8);
        uint8_t* b2 = (uint8_t*)b_sel.get((size_t)n * 2 + 2);
        uint8_t* b3 = b2 + n + 1;
        k_parent_keys<<<(n + 255) / 256, 256>>>(pairs, n, nullptr, R.parent, k1);
        tmp = 0; cub::DeviceRadixSort::SortPairs(nullptr, tmp, k1, k2, hit, b2, n);
        cub::DeviceRadixSort::SortPairs(b_tmp.get(tmp), tmp, k1, k2, hit, b2, n);
        tmp = 0; cub::DeviceReduce::ReduceByKey(nullptr, tmp, k2, k1, b2, b3, nsel, OrU8{}, n);
        cub::DeviceReduce::ReduceByKey(b_tmp.get(tmp), tmp, k2, k1, b2, b3, nsel, OrU8{}, n);
        int nu = 0; CK(cudaMemcpy(&nu, nsel, 4, cudaMemcpyDeviceToHost));
        k_is_one_u8<<<(nu + 255) / 256, 256>>>(b3, nu, b2);
        tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, k1, b2, k2, nsel, nu);
        cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, k1, b2, k2, nsel, nu);
        CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
        k_unpack_keys<<<(nh + 255) / 256, 256>>>(k2, nh, opairs);
      } else {
        if (er) k_is_one_u8<<<(n + 255) / 256, 256>>>(hit, n, hit);
        tmp = 0; cub::DeviceSelect::Flagged(nullptr, tmp, pairs, hit, opairs, nsel, (int)n);
        cub::DeviceSelect::Flagged(b_tmp.get(tmp), tmp, pairs, hit, opairs, nsel, (int)n);
        CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
      }
      if (!er && R.parent && nh) {   // (point, piece) hits -> unique (point, parent)
        unsigned long long* k1 = (unsigned long long*)b_keys.get((size_t)nh * 8);
        unsigned long long* k2 = (unsigned long long*)b_keys2.get((size_t)nh * 8);
        k_parent_keys<<<(nh + 255) / 256, 256>>>(opairs, nh, nullptr, R.parent, k1);
        tmp = 0; cub::DeviceRadixSort::SortKeys(nullptr, tmp, k1, k2, nh);
        cub::DeviceRadixSort::SortKeys(b_tmp.get(tmp), tmp, k1, k2, nh);
        tmp = 0; cub::DeviceSelect::Unique(nullptr, tmp, k2, k1, nsel, nh);
        cub::DeviceSelect::Unique(b_tmp.get(tmp), tmp, k2, k1, nsel, nh);
        CK(cudaMemcpy(&nh, nsel, 4, cudaMemcpyDeviceToHost));
        k_unpack_keys<<<(nh + 255) / 256, 256>>>(k1, nh, opairs);
      }
      st.compact_ms += tm.stop();
      st.hits += nh;
      if (!nh) continue;
      tm.start();
      uint2* hp = (uint2*)h_pairs.get((size_t)nh * sizeof(uint2) + 8);
      CK(cudaMemcpy(hp, opairs, (size_t)nh * sizeof(uint2), cudaMemcpyDeviceToHost));
      st.copy_ms += tm.stop();
      cb(hp, (size_t)nh);
    }
  }
  return st;
}
