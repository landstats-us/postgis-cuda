#pragma once
// Parallel layer loader: COPY (SELECT id::text, geom ...) TO STDOUT (FORMAT binary) over N libpq connections,
// one chunk per (leaf partition, block range) via TID range scans. Geometry arrives as EWKB (geometry_send).
#include "../engine/gset.h"
#include <libpq-fe.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>

// Packed text ids (157M uuids as std::string would cost ~12 GB of heap overhead)
struct IdStore {
  std::string buf; std::vector<uint64_t> off{0};
  void add(const char* p, size_t n) { buf.append(p, n); off.push_back(buf.size()); }
  size_t size() const { return off.size() - 1; }
  const char* data(size_t i) const { return buf.data() + off[i]; }
  size_t len(size_t i) const { return off[i + 1] - off[i]; }
  void append(const IdStore& o) {
    uint64_t base = buf.size(); buf += o.buf;
    for (size_t i = 1; i < o.off.size(); i++) off.push_back(base + o.off[i]);
  }
};

struct LoadSpec {
  std::string table, id_col = "id", geom_col = "geom", filter;   // table mode
  std::string sql;                                                // query mode (single stream)
  int threads = 16;
  uint32_t pages_per_chunk = 16384;                              // 128 MB of heap per COPY
};

struct LoadedData {
  bool is_point = false;
  HostSet poly; HostPoints pts; IdStore ids;
  uint64_t rows = 0, skipped_null = 0, skipped_type = 0, skipped_empty = 0, chunks = 0;
  double seconds = 0;
};

namespace loader_detail {

struct Chunk { std::string sql; };

struct ChunkOut {
  std::vector<uint64_t> gro{0}, rvo{0}; std::vector<double> xy;   // polygons
  std::vector<double> pxy;                                         // points
  IdStore ids;
  uint64_t rows = 0, skipped_null = 0, skipped_type = 0, skipped_empty = 0, polys = 0, points = 0;
};

struct Rd {
  const uint8_t* p; const uint8_t* e; bool le = true;
  bool ok(size_t n) const { return (size_t)(e - p) >= n; }
  uint32_t u32() { uint32_t v; memcpy(&v, p, 4); p += 4; return le ? v : __builtin_bswap32(v); }
  double f64() { uint64_t v; memcpy(&v, p, 8); p += 8; if (!le) v = __builtin_bswap64(v); double d; memcpy(&d, &v, 8); return d; }
};

// Twice the signed area of a closed ring, accurate enough to orient sliver rings: coordinates relative to the first
// vertex (exact differences), products with their rounding errors (fma), compensated (Neumaier) summation. On absolute
// lon/lat the products cancel catastrophically and a 1e-14 deg^2 ring can come out with the wrong sign.
static double ring_area2(const double* xy, size_t n) {
  if (n < 4) return 0;
  const double x0 = xy[0], y0 = xy[1];
  double s = 0, c = 0;
  auto add = [&](double v) { double t = s + v; c += std::fabs(s) >= std::fabs(v) ? (s - t) + v : (v - t) + s; s = t; };
  for (size_t i = 1; i + 2 < n; i++) {
    double ax = xy[2 * i] - x0, ay = xy[2 * i + 1] - y0, bx = xy[2 * i + 2] - x0, by = xy[2 * i + 3] - y0;
    double p = ax * by, q = ay * bx;
    add(p); add(-q); add(std::fma(ax, by, -p)); add(-std::fma(ay, bx, -q));
  }
  return s + c;
}

// Append the polygons of one EWKB geometry as rings of a single output geometry (shells CCW, holes CW).
// Returns false on malformed input; sets kind: 0 none/empty, 1 polygonal, 2 point.
static bool parse_ewkb(Rd& r, ChunkOut& o, std::vector<double>& tmp, size_t& nrings_added, int& kind) {
  if (!r.ok(5)) return false;
  r.le = (*r.p++ == 1);
  uint32_t t = r.u32();
  bool hz = t & 0x80000000u, hm = t & 0x40000000u;
  if (t & 0x20000000u) { if (!r.ok(4)) return false; r.u32(); }
  uint32_t base = t & 0x0FFFFFFFu;
  if (base >= 1000) { uint32_t iso = base / 1000; base %= 1000; hz = hz || iso == 1 || iso == 3; hm = hm || iso == 2 || iso == 3; }
  int nd = 2 + hz + hm;
  switch (base) {
    case 1: {   // point
      if (!r.ok(8 * nd)) return false;
      double x = r.f64(), y = r.f64(); for (int k = 2; k < nd; k++) r.f64();
      if (std::isnan(x) || std::isnan(y)) return true;   // POINT EMPTY
      o.pxy.push_back(x); o.pxy.push_back(y); kind = 2; return true;
    }
    case 3: {   // polygon
      if (!r.ok(4)) return false;
      uint32_t nr = r.u32();
      for (uint32_t k = 0; k < nr; k++) {
        if (!r.ok(4)) return false;
        uint32_t np = r.u32();
        if (!r.ok((size_t)np * 8 * nd)) return false;
        tmp.resize(2 * (size_t)np);
        for (uint32_t i = 0; i < np; i++) { tmp[2 * i] = r.f64(); tmp[2 * i + 1] = r.f64(); for (int d = 2; d < nd; d++) r.f64(); }
        if (np < 4) continue;
        double a = ring_area2(tmp.data(), np);
        bool want_ccw = (k == 0);
        if ((a > 0) != want_ccw) for (uint32_t i = 0; i < np / 2; i++) { std::swap(tmp[2 * i], tmp[2 * (np - 1 - i)]); std::swap(tmp[2 * i + 1], tmp[2 * (np - 1 - i) + 1]); }
        o.xy.insert(o.xy.end(), tmp.begin(), tmp.end());
        o.rvo.push_back(o.xy.size() / 2);
        nrings_added++;
      }
      if (kind == 0) kind = 1;
      return true;
    }
    case 6: case 7: {   // multipolygon, geometrycollection (polygonal members only)
      if (!r.ok(4)) return false;
      uint32_t ng = r.u32();
      for (uint32_t k = 0; k < ng; k++) {
        int sub = 0;
        if (!parse_ewkb(r, o, tmp, nrings_added, sub)) return false;
        if (sub == 2) return false;   // mixed points and polygons: unsupported
        if (sub == 1) kind = 1;
      }
      return true;
    }
    default: return false;          // lines etc.
  }
}

// Parse a binary COPY stream incrementally.
struct CopyParser {
  std::string acc; size_t pos = 0; bool header = false, done = false;
  std::vector<double> tmp;
  bool feed(const char* d, int n, ChunkOut& o) {
    acc.append(d, n);
    const uint8_t* b = (const uint8_t*)acc.data();
    size_t len = acc.size();
    if (!header) {
      if (len < 19) return true;
      if (memcmp(b, "PGCOPY\n\377\r\n\0", 11)) return false;
      uint32_t ext; memcpy(&ext, b + 15, 4); ext = __builtin_bswap32(ext);
      if (len < 19 + ext) return true;
      pos = 19 + ext; header = true;
    }
    for (;;) {
      if (len - pos < 2) break;
      int16_t nf; memcpy(&nf, b + pos, 2); nf = (int16_t)__builtin_bswap16((uint16_t)nf);
      if (nf == -1) { done = true; pos += 2; break; }
      // need both fields complete
      size_t q = pos + 2; int32_t l0, l1;
      if (len - q < 4) break;
      memcpy(&l0, b + q, 4); l0 = (int32_t)__builtin_bswap32((uint32_t)l0); q += 4;
      if (l0 > 0 && len - q < (size_t)l0) break;
      size_t id_at = q; q += l0 > 0 ? l0 : 0;
      if (len - q < 4) break;
      memcpy(&l1, b + q, 4); l1 = (int32_t)__builtin_bswap32((uint32_t)l1); q += 4;
      if (l1 > 0 && len - q < (size_t)l1) break;
      size_t g_at = q; q += l1 > 0 ? l1 : 0;
      o.rows++;
      if (l1 < 0 || l0 < 0) o.skipped_null++;
      else {
        Rd r{b + g_at, b + g_at + l1};
        size_t nr = 0; int kind = 0;
        size_t xy0 = o.xy.size(), rvo0 = o.rvo.size(), p0 = o.pxy.size();
        if (!parse_ewkb(r, o, tmp, nr, kind) || (o.polys && kind == 2) || (o.points && kind == 1)) {
          o.xy.resize(xy0); o.rvo.resize(rvo0); o.pxy.resize(p0); o.skipped_type++;
        } else if (kind == 0 || (kind == 1 && nr == 0)) {
          o.xy.resize(xy0); o.rvo.resize(rvo0); o.skipped_empty++;
        } else {
          if (kind == 1) { o.gro.push_back(o.rvo.size() - 1); o.polys++; } else o.points++;
          o.ids.add((const char*)b + id_at, l0);
        }
      }
      pos = q;
    }
    if (pos > (1u << 24)) { acc.erase(0, pos); pos = 0; }
    return true;
  }
};

static std::string q_ident(PGconn* c, const std::string& s) {
  char* e = PQescapeIdentifier(c, s.data(), s.size()); std::string r = e ? e : ""; PQfreemem(e); return r;
}

static std::string run_copy(PGconn* c, const std::string& sql, ChunkOut& o) {
  PGresult* res = PQexec(c, ("COPY (" + sql + ") TO STDOUT (FORMAT binary)").c_str());
  if (PQresultStatus(res) != PGRES_COPY_OUT) { std::string e = PQerrorMessage(c); PQclear(res); return e; }
  PQclear(res);
  CopyParser cp; char* buf;
  int n;
  while ((n = PQgetCopyData(c, &buf, 0)) > 0) {
    bool ok = cp.feed(buf, n, o); PQfreemem(buf);
    if (!ok) return "malformed COPY stream";
  }
  if (n == -2) return PQerrorMessage(c);
  res = PQgetResult(c);
  std::string err = PQresultStatus(res) == PGRES_COMMAND_OK ? "" : PQerrorMessage(c);
  PQclear(res);
  while ((res = PQgetResult(c))) PQclear(res);
  if (err.empty() && (o.polys && o.points)) err = "layer mixes points and polygons";
  return err;
}

}  // namespace loader_detail

// Load a layer. Returns an error string ("" on success).
inline std::string load_layer(const std::string& conninfo, const LoadSpec& spec, LoadedData& out) {
  using namespace loader_detail;
  auto t0 = std::chrono::steady_clock::now();
  PGconn* c0 = PQconnectdb(conninfo.c_str());
  if (PQstatus(c0) != CONNECTION_OK) { std::string e = PQerrorMessage(c0); PQfinish(c0); return "connect: " + e; }
  std::vector<Chunk> chunks;
  if (!spec.sql.empty()) {
    chunks.push_back({"SELECT (q.id)::text, q.geom FROM (" + spec.sql + ") q(id, geom)"});
  } else {
    // leaf relations with storage and their size in blocks
    const char* params[1] = {spec.table.c_str()};
    PGresult* r = PQexecParams(c0,
      "WITH RECURSIVE t(oid) AS (SELECT $1::regclass::oid UNION ALL SELECT i.inhrelid FROM pg_inherits i JOIN t ON i.inhparent = t.oid) "
      "SELECT c.oid::regclass::text, c.relkind, (pg_relation_size(c.oid) / current_setting('block_size')::int)::bigint "
      "FROM t JOIN pg_class c ON c.oid = t.oid WHERE c.relkind IN ('r','m','f') ORDER BY c.oid", 1, nullptr, params, nullptr, nullptr, 0);
    if (PQresultStatus(r) != PGRES_TUPLES_OK) { std::string e = PQerrorMessage(c0); PQclear(r); PQfinish(c0); return e; }
    std::string idc = q_ident(c0, spec.id_col), gc = q_ident(c0, spec.geom_col);
    std::string where = spec.filter.empty() ? "" : " AND (" + spec.filter + ")";
    for (int i = 0; i < PQntuples(r); i++) {
      std::string rel = PQgetvalue(r, i, 0); char kind = PQgetvalue(r, i, 1)[0]; long long pages = atoll(PQgetvalue(r, i, 2));
      std::string sel = "SELECT (" + idc + ")::text, " + gc + " FROM ONLY " + rel + " WHERE true" + where;
      if (kind == 'f' || pages <= (long long)spec.pages_per_chunk) { chunks.push_back({sel}); continue; }
      for (long long b0 = 0; b0 < pages; b0 += spec.pages_per_chunk) {
        long long b1 = b0 + spec.pages_per_chunk;
        std::string tr = " AND ctid >= '(" + std::to_string(b0) + ",0)'::tid" + (b1 < pages ? " AND ctid < '(" + std::to_string(b1) + ",0)'::tid" : "");
        chunks.push_back({sel + tr});
      }
    }
    PQclear(r);
  }
  PQfinish(c0);
  // fetch chunks in parallel, keep results in chunk order (deterministic)
  std::vector<ChunkOut> outs(chunks.size());
  std::atomic<size_t> next{0}; std::mutex em; std::string err;
  int nt = std::max(1, std::min<int>(spec.threads, (int)chunks.size()));
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) th.emplace_back([&] {
    PGconn* c = PQconnectdb(conninfo.c_str());
    if (PQstatus(c) != CONNECTION_OK) { std::lock_guard<std::mutex> g(em); err = PQerrorMessage(c); PQfinish(c); return; }
    PQexec(c, "SET statement_timeout = 0");
    for (size_t i; (i = next++) < chunks.size();) {
      { std::lock_guard<std::mutex> g(em); if (!err.empty()) break; }
      std::string e = run_copy(c, chunks[i].sql, outs[i]);
      if (!e.empty()) { std::lock_guard<std::mutex> g(em); if (err.empty()) err = e + " [in: " + chunks[i].sql.substr(0, 200) + "]"; break; }
    }
    PQfinish(c);
  });
  for (auto& t : th) t.join();
  if (!err.empty()) return err;
  // merge
  uint64_t polys = 0, points = 0, nr = 0, nv = 0, npts = 0;
  for (auto& o : outs) {
    polys += o.polys; points += o.points; nr += o.rvo.size() - 1; nv += o.xy.size() / 2; npts += o.pxy.size() / 2;
    out.rows += o.rows; out.skipped_null += o.skipped_null; out.skipped_type += o.skipped_type; out.skipped_empty += o.skipped_empty;
  }
  if (polys && points) return "layer mixes points and polygons";
  out.is_point = points > 0;
  out.chunks = chunks.size();
  if (out.is_point) {
    out.pts.n = npts; out.pts.xy.reserve(2 * npts);
    for (auto& o : outs) { out.pts.xy.insert(out.pts.xy.end(), o.pxy.begin(), o.pxy.end()); out.ids.append(o.ids); std::vector<double>().swap(o.pxy); }
  } else {
    HostSet& h = out.poly; h.n = polys; h.nr = nr; h.nv = nv;
    h.gro.reserve(polys + 1); h.rvo.reserve(nr + 1); h.xy.resize(2 * nv);
    h.gro.push_back(0); h.rvo.push_back(0);
    uint64_t rb = 0, vb = 0;
    for (auto& o : outs) {
      for (size_t i = 1; i < o.gro.size(); i++) h.gro.push_back(rb + o.gro[i]);
      for (size_t i = 1; i < o.rvo.size(); i++) h.rvo.push_back(vb + o.rvo[i]);
      memcpy(h.xy.data() + 2 * vb, o.xy.data(), o.xy.size() * 8);
      rb += o.rvo.size() - 1; vb += o.xy.size() / 2;
      out.ids.append(o.ids);
      std::vector<double>().swap(o.xy); std::vector<uint64_t>().swap(o.rvo);
    }
  }
  out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return "";
}
