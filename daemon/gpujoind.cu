// gpujoind: GPU spatial join daemon. Owns one GPU, keeps named layers resident, serves the gpu_spatial
// Postgres extension over a unix socket (and optionally TCP). See protocol.h for the wire format.
//
//   gpujoind --conninfo "host=... dbname=... user=..." [--socket PATH] [--tcp HOST:PORT] [--threads N]
//
// Select the GPU with CUDA_VISIBLE_DEVICES.
#include "../engine/engine.cuh"
#include "../engine/buffer.cuh"
#include "../engine/subdivide.h"
#include "../engine/dedup.h"
#include "loader.h"
#include "protocol.h"
#include <map>
#include <fstream>
#include <omp.h>
#include <memory>
#include <csignal>
#include <cstdarg>
#include <limits>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <poll.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

static std::string g_conninfo;
static int g_threads = 16;
static std::string g_token;   // shared secret; required when non-empty
static std::mutex g_gpu;   // one GPU job at a time

static void logf(const char* fmt, ...) {
  char ts[32]; time_t now = time(nullptr); strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", localtime(&now));
  va_list ap; va_start(ap, fmt);
  fprintf(stderr, "[%s] ", ts); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
  va_end(ap);
}

// ------------------------------------------------------------------------------------------------- JSON
static std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
    else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
    else o += (char)c;
  }
  return o + "\"";
}
struct J {
  std::string s = "{"; bool first = true;
  J& k(const char* key) { if (!first) s += ','; first = false; s += jstr(key) + ":"; return *this; }
  J& v(const std::string& x) { s += jstr(x); return *this; }
  J& n(double x) { char b[64]; snprintf(b, sizeof b, "%.6g", x); s += b; return *this; }
  J& u(uint64_t x) { s += std::to_string(x); return *this; }
  J& b(bool x) { s += x ? "true" : "false"; return *this; }
  J& raw(const std::string& x) { s += x; return *this; }
  std::string done() { return s + "}"; }
};

// ------------------------------------------------------------------------------------------------- layers
struct Layer {
  std::string name, kind, source; int subdiv = 0; bool dedup = true;
  IdStore ids; uint64_t rows = 0, skipped = 0;
  bool deduped = false; std::vector<uint64_t> moff; std::vector<uint32_t> midx; uint64_t n_distinct = 0;
  std::vector<uint32_t> parent_h;       // pieces -> distinct geometry (DSet.h_parent points here)
  DSet d; Bvh bvh; PSetD pts;
  uint64_t gpu_geoms = 0, verts = 0; size_t gpu_bytes = 0;
  double copy_s = 0, prep_s = 0, upload_s = 0;
  template <class F> void members(uint32_t g, F&& f) const {
    if (!deduped) { f(g); return; }
    for (uint64_t k = moff[g]; k < moff[g + 1]; k++) f(midx[k]);
  }
  ~Layer() {
    std::lock_guard<std::mutex> g(g_gpu);
    if (d.xy) free_set(d);
    if (bvh.nbb) free_bvh(bvh);
    if (pts.xy) free_points(pts);
  }
  std::string json() const {
    return J().k("layer").v(name).k("kind").v(kind).k("source").v(source)
        .k("rows").u(rows).k("skipped").u(skipped).k("distinct").u(n_distinct).k("gpu_geoms").u(gpu_geoms).k("vertices").u(verts)
        .k("subdivide").u(subdiv).k("dedup").b(deduped).k("gpu_mb").n(gpu_bytes / 1048576.0)
        .k("copy_s").n(copy_s).k("prepare_s").n(prep_s).k("upload_s").n(upload_s).done();
  }
};
static std::mutex g_reg;
static std::map<std::string, std::shared_ptr<Layer>> g_layers;

static std::shared_ptr<Layer> get_layer(const std::string& n) {
  std::lock_guard<std::mutex> g(g_reg);
  auto it = g_layers.find(n);
  return it == g_layers.end() ? nullptr : it->second;
}

static std::string build_layer(const std::string& name, LoadSpec spec, int subdiv, bool dedup, const std::string& source) {
  spec.threads = g_threads;
  LoadedData ld;
  logf("load %s: start (%s)", name.c_str(), source.c_str());
  std::string err = load_layer(g_conninfo, spec, ld);
  if (!err.empty()) throw std::runtime_error("load failed: " + err);
  auto L = std::make_shared<Layer>();
  L->name = name; L->subdiv = subdiv; L->source = source; L->copy_s = ld.seconds;
  L->rows = ld.ids.size(); L->skipped = ld.skipped_null + ld.skipped_type + ld.skipped_empty;
  L->ids = std::move(ld.ids);
  logf("load %s: %llu rows in %.1fs over %llu chunks (skipped null=%llu type=%llu empty=%llu)", name.c_str(), (unsigned long long)L->rows,
       ld.seconds, (unsigned long long)ld.chunks, (unsigned long long)ld.skipped_null, (unsigned long long)ld.skipped_type, (unsigned long long)ld.skipped_empty);
  auto t0 = Clock::now();
  if (ld.is_point) {
    L->kind = "point"; L->n_distinct = L->rows; L->gpu_geoms = L->rows; L->verts = L->rows;
    std::lock_guard<std::mutex> g(g_gpu);
    auto t1 = Clock::now();
    L->pts = upload_points(ld.pts.xy); L->gpu_bytes = L->pts.bytes;
    CK(cudaDeviceSynchronize()); L->upload_s = secs_since(t1);
  } else {
    L->kind = "polygon";
    if (L->rows == 0) throw std::runtime_error("layer has no polygonal rows");
    Dedup dd; const HostSet* base = &ld.poly;
    if (dedup) {
      dd = dedup_set(ld.poly);
      L->deduped = true; L->moff = std::move(dd.moff); L->midx = std::move(dd.midx); base = &dd.u;
      HostSet().gro.swap(ld.poly.gro); std::vector<double>().swap(ld.poly.xy); std::vector<uint64_t>().swap(ld.poly.rvo);
    }
    L->n_distinct = base->n;
    HostSet pieces; std::vector<uint8_t> syn;
    if (subdiv > 0) { pieces = subdivide_set(*base, subdiv, L->parent_h, syn); base = &pieces; }
    L->gpu_geoms = base->n; L->verts = base->nv;
    L->prep_s = secs_since(t0);
    // device memory check: vertices + per-geometry arrays + BVH, with headroom
    size_t need = base->nv * 16 + base->nr * 8 + base->n * (16 + 8 + 8 + 4 + 16 + 4 + 2 * 20 + 8) + (size_t)(base->nv / 8);
    std::lock_guard<std::mutex> g(g_gpu);
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    if (need + (2ull << 30) > fr) throw std::runtime_error("not enough GPU memory: need ~" + std::to_string(need >> 20) + " MB, free " + std::to_string(fr >> 20) + " MB");
    auto t1 = Clock::now();
    L->d = subdiv > 0 ? upload_set(*base, &L->parent_h, &syn) : upload_set(*base);
    L->bvh = build_bvh(L->d);
    CK(cudaDeviceSynchronize());
    L->gpu_bytes = L->d.bytes + L->bvh.bytes; L->upload_s = secs_since(t1);
  }
  std::string js = L->json();
  logf("load %s: ready %s", name.c_str(), js.c_str());
  std::shared_ptr<Layer> old;
  { std::lock_guard<std::mutex> g(g_reg); auto& slot = g_layers[name]; old = slot; slot = L; }
  return js;
}

// ------------------------------------------------------------------------------------------------- io
struct Conn {
  int fd;
  std::atomic<bool> broken{false};
  bool streaming = false;   // a result stream has started: errors must be sent as GSP_FRAME_ERR
  bool read_full(void* p, size_t n) {
    char* c = (char*)p;
    while (n) { ssize_t r = ::read(fd, c, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return false; } c += r; n -= r; }
    return true;
  }
  bool write_full(const void* p, size_t n) {
    const char* c = (const char*)p;
    while (n && !broken) {
      ssize_t r = ::send(fd, c, n, MSG_NOSIGNAL);
      if (r <= 0) { if (r < 0 && errno == EINTR) continue; broken = true; return false; }
      c += r; n -= r;
    }
    return !broken;
  }
  bool u32(uint32_t v) { return write_full(&v, 4); }
  bool str(const std::string& s) { return u32((uint32_t)s.size()) && write_full(s.data(), s.size()); }
  void ok_json(const std::string& j) { u32(GSP_OK); str(j); }
  void error(const std::string& m) { u32(GSP_ERR); str(m); }
};

// Row frames for streaming results. Rows are encoded in parallel into independent buffers of complete frames
// (one per slice of the result batch) and sent in order.
struct FrameBuf {
  std::string out; size_t fstart = 0; uint32_t cnt = 0; uint64_t rows = 0;
  void begin() { fstart = out.size(); out.append(4, '\0'); cnt = 0; }
  void id(const char* p, size_t l) { uint16_t L = (uint16_t)std::min<size_t>(l, 65535); out.append((const char*)&L, 2); out.append(p, L); }
  void f64(double x) { out.append((const char*)&x, 8); }
  void u32(uint32_t x) { out.append((const char*)&x, 4); }
  void row_done() { cnt++; rows++; if (out.size() - fstart >= (1u << 20)) { close(); begin(); } }
  void close() { memcpy(&out[fstart], &cnt, 4); }
  void finish() { if (cnt) close(); else out.resize(fstart); }
};

template <class Enc>   // enc(size_t i, FrameBuf&) encodes the rows of item i
static uint64_t send_parallel(Conn& c, size_t n, Enc&& enc) {
  if (!n || c.broken) return 0;
  int T = omp_get_max_threads();
  size_t nch = std::min<size_t>(n, (size_t)T * 4);
  std::vector<FrameBuf> fb(nch);
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t k = 0; k < (int64_t)nch; k++) {
    size_t i0 = n * k / nch, i1 = n * (k + 1) / nch;
    fb[k].out.reserve((i1 - i0) * 96);
    fb[k].begin();
    for (size_t i = i0; i < i1; i++) enc(i, fb[k]);
    fb[k].finish();
  }
  uint64_t rows = 0;
  for (auto& f : fb) { if (!c.write_full(f.out.data(), f.out.size())) break; rows += f.rows; }
  return rows;
}
static void end_stream(Conn& c, const std::string& stats) { c.u32(0); c.str(stats); }

// ------------------------------------------------------------------------------------------------- ops
static void op_join(Conn& c, const std::vector<std::string>& a) {
  if (a.size() < 4) throw std::runtime_error("JOIN needs layer_a, layer_b, want_area, self_pairs[, mode]");
  bool stats_only = a.size() > 4 && a[4] == "stats";   // run the join, return only counts / area sum
  auto A = get_layer(a[0]), B = get_layer(a[1]);
  if (!A) throw std::runtime_error("no such layer: " + a[0]);
  if (!B) throw std::runtime_error("no such layer: " + a[1]);
  bool want_area = a[2] == "1" || a[2] == "t" || a[2] == "true";
  bool distinct = a[3] == "distinct";
  if (a[3] != "all" && a[3] != "distinct") throw std::runtime_error("self_pairs must be 'all' or 'distinct'");
  bool same = A.get() == B.get();
  if (A->kind == "point" && B->kind == "point") throw std::runtime_error("point x point joins are not supported");
  bool swapped = false;
  if (A->kind == "polygon" && B->kind == "point") { std::swap(A, B); swapped = true; }   // run as point x polygon
  std::lock_guard<std::mutex> g(g_gpu);
  c.u32(GSP_OK); c.streaming = true;
  JoinOpts jo; jo.cancel = &c.broken;
  auto t0 = Clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Layer& LA = swapped ? *B : *A; const Layer& LB = swapped ? *A : *B;   // output order (a_id, b_id)
  uint64_t total = 0; double host_s = 0, area_sum = 0;
  std::string stats;
  auto msize = [](const Layer& L, uint32_t g) -> uint64_t { return L.deduped ? L.moff[g + 1] - L.moff[g] : 1; };
  if (A->kind == "point") {
    PointJoinStats st;
    {
      st = join_points_stream(A->pts, B->d, B->bvh, jo, [&](const uint2* p, const uint8_t*, size_t n) {
        auto th = Clock::now();
        if (stats_only) { for (size_t i = 0; i < n; i++) total += msize(*B, p[i].y); host_s += secs_since(th); return; }
        total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
          B->members(p[i].y, [&](uint32_t rb) {
            uint32_t ra = p[i].x;
            if (swapped) std::swap(ra, rb);
            f.id(LA.ids.data(ra), LA.ids.len(ra)); f.id(LB.ids.data(rb), LB.ids.len(rb)); f.f64(want_area ? 0.0 : nan); f.row_done(); });
        });
        host_s += secs_since(th);
      });
    }
    stats = J().k("kind").v("point_in_polygon").k("candidates").u(st.candidates).k("gpu_hits").u(st.hits).k("rows").u(total)
        .k("traverse_ms").n(st.traverse_ms).k("refine_ms").n(st.refine_ms).k("host_encode_send_s").n(host_s).k("total_s").n(secs_since(t0)).done();
    if (stats_only) stats = stats.substr(0, stats.size() - 1) + ",\"stats_only\":true" + (want_area ? ",\"area_sum\":" + J().n(area_sum).s.substr(1) : std::string()) + "}";
  } else {
    RefineParams rp; rp.want_area = want_area ? 1 : 0;
    JoinStats st;
    jo.upper = same;   // self-join: each unordered pair once, mirrored below
    {
      st = join_stream(A->d, B->d, B->bvh, rp, jo, [&](const uint2* p, const double* ar, const uint32_t*, size_t n) {
        auto th = Clock::now();
        if (stats_only) {
          uint64_t t = 0; double as = 0;
#pragma omp parallel for reduction(+ : t, as)
          for (int64_t i = 0; i < (int64_t)n; i++) {
            uint64_t ma = msize(*A, p[i].x), mb = msize(*B, p[i].y), k;
            if (!same) k = ma * mb;
            else if (p[i].x == p[i].y) k = distinct ? ma * (ma - 1) / 2 : ma * ma;
            else k = distinct ? ma * mb : 2 * ma * mb;
            t += k; if (ar) as += ar[i] * (double)k;
          }
          total += t; area_sum += as; host_s += secs_since(th); return;
        }
        total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
          double area = ar ? ar[i] : nan;
          bool diag = same && p[i].x == p[i].y;
          A->members(p[i].x, [&](uint32_t ra) { B->members(p[i].y, [&](uint32_t rb) {
            if (!same) { f.id(A->ids.data(ra), A->ids.len(ra)); f.id(B->ids.data(rb), B->ids.len(rb)); f.f64(area); f.row_done(); return; }
            if (distinct) {
              if (ra == rb || (diag && ra > rb)) return;
              uint32_t lo = std::min(ra, rb), hi = std::max(ra, rb);
              f.id(A->ids.data(lo), A->ids.len(lo)); f.id(A->ids.data(hi), A->ids.len(hi)); f.f64(area); f.row_done();
              return;
            }
            f.id(A->ids.data(ra), A->ids.len(ra)); f.id(A->ids.data(rb), A->ids.len(rb)); f.f64(area); f.row_done();
            if (!diag) { f.id(A->ids.data(rb), A->ids.len(rb)); f.id(A->ids.data(ra), A->ids.len(ra)); f.f64(area); f.row_done(); }
          }); });
        });
        host_s += secs_since(th);
      });
    }
    stats = J().k("kind").v("polygon_polygon").k("candidates").u(st.candidates).k("gpu_hits").u(st.hits).k("rows").u(total).k("heavy").u(st.heavy)
        .k("traverse_ms").n(st.traverse_ms).k("refine_ms").n(st.refine_ms).k("compact_ms").n(st.compact_ms).k("d2h_ms").n(st.copy_ms)
        .k("host_encode_send_s").n(host_s).k("total_s").n(secs_since(t0)).done();
    if (stats_only) stats = stats.substr(0, stats.size() - 1) + ",\"stats_only\":true" + (want_area ? ",\"area_sum\":" + J().n(area_sum).s.substr(1) : std::string()) + "}";
  }
  if (!c.broken) end_stream(c, stats);
  logf("join %s x %s area=%d: %s%s", a[0].c_str(), a[1].c_str(), (int)want_area, stats.c_str(), c.broken ? " (client went away)" : "");
}

static void op_cluster(Conn& c, const std::vector<std::string>& a) {
  if (a.size() < 1) throw std::runtime_error("CLUSTER needs layer");
  auto L = get_layer(a[0]);
  if (!L) throw std::runtime_error("no such layer: " + a[0]);
  if (L->kind != "polygon") throw std::runtime_error("cluster needs a polygon layer");
  std::lock_guard<std::mutex> g(g_gpu);
  auto t0 = Clock::now();
  std::vector<uint32_t> root; ClusterStats cs;
  JoinOpts jo; jo.cancel = &c.broken;
  {
    RefineParams rp; rp.want_area = 0;
    root = cluster_intersecting(L->d, (uint32_t)L->n_distinct, L->bvh, rp, jo, &cs);
  }
  // dense cluster numbers in order of first row (like ST_ClusterIntersectingWin numbering from 0)
  std::vector<uint32_t> row_root(L->rows), dense(root.size(), UINT32_MAX), label(L->rows);
  for (uint32_t gi = 0; gi < L->n_distinct; gi++) L->members(gi, [&](uint32_t r) { row_root[r] = root[gi]; });
  uint32_t k = 0;
  for (uint64_t r = 0; r < L->rows; r++) { uint32_t rt = row_root[r]; if (dense[rt] == UINT32_MAX) dense[rt] = k++; label[r] = dense[rt]; }
  c.u32(GSP_OK); c.streaming = true;
  uint64_t total = send_parallel(c, L->rows, [&](size_t r, FrameBuf& f) { f.id(L->ids.data(r), L->ids.len(r)); f.u32(label[r]); f.row_done(); });
  std::string stats = J().k("kind").v("cluster_intersecting").k("rows").u(total).k("clusters").u(k).k("candidates").u(cs.join.candidates)
      .k("edges").u(cs.join.hits).k("traverse_ms").n(cs.join.traverse_ms).k("refine_ms").n(cs.join.refine_ms).k("union_find_ms").n(cs.uf_ms)
      .k("total_s").n(secs_since(t0)).done();
  if (!c.broken) end_stream(c, stats);
  logf("cluster %s: %s", a[0].c_str(), stats.c_str());
}

// ST_Relate for intersecting pairs: rows (a_id, b_id, DE-9IM), optionally filtered by a DE-9IM pattern.
static void op_relate(Conn& c, const std::vector<std::string>& a) {
  if (a.size() < 4) throw std::runtime_error("RELATE needs layer_a, layer_b, pattern, self_pairs");
  auto A = get_layer(a[0]), B = get_layer(a[1]);
  if (!A) throw std::runtime_error("no such layer: " + a[0]);
  if (!B) throw std::runtime_error("no such layer: " + a[1]);
  const std::string pat = a[2];
  if (!pat.empty()) {
    if (pat.size() != 9) throw std::runtime_error("DE-9IM pattern must have 9 characters");
    for (char ch : pat) if (!strchr("TtFf*012", ch)) throw std::runtime_error("bad DE-9IM pattern character '" + std::string(1, ch) + "'");
  }
  bool distinct = a[3] == "distinct";
  if (a[3] != "all" && a[3] != "distinct") throw std::runtime_error("self_pairs must be 'all' or 'distinct'");
  bool same = A.get() == B.get();
  if (A->kind == "point" && B->kind == "point") throw std::runtime_error("point x point relate is not supported");
  bool swapped = false;
  if (A->kind == "polygon" && B->kind == "point") { std::swap(A, B); swapped = true; }
  std::lock_guard<std::mutex> g(g_gpu);
  c.u32(GSP_OK); c.streaming = true;
  JoinOpts jo; jo.cancel = &c.broken;
  auto t0 = Clock::now();
  uint64_t total = 0, cand = 0, hits = 0; double host_s = 0;
  auto put = [&](FrameBuf& f, const Layer& LA, uint32_t ra, const Layer& LB, uint32_t rb, uint32_t code) {
    if (!pat.empty() && !de9im_match(code, pat.c_str())) return;
    char m[9]; de9im_str(code, m);
    f.id(LA.ids.data(ra), LA.ids.len(ra)); f.id(LB.ids.data(rb), LB.ids.len(rb)); f.out.append(m, 9); f.row_done();
  };
  if (A->kind == "point") {
    PointJoinStats st = join_points_stream(A->pts, B->d, B->bvh, jo, [&](const uint2* p, const uint8_t* wh, size_t n) {
      auto th = Clock::now();
      total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
        uint32_t code = wh[i] == 2 ? DE9IM_PT_BOUNDARY : DE9IM_PT_INTERIOR;   // relate(point, polygon)
        B->members(p[i].y, [&](uint32_t rb) {
          if (swapped) put(f, *B, rb, *A, p[i].x, de9im_transpose(code)); else put(f, *A, p[i].x, *B, rb, code); });
      });
      host_s += secs_since(th);
    });
    cand = st.candidates; hits = st.hits;
  } else {
    RefineParams rp; rp.want_area = 0; rp.want_relate = 1;
    jo.upper = same;
    JoinStats st = join_stream(A->d, B->d, B->bvh, rp, jo, [&](const uint2* p, const double*, const uint32_t* code, size_t n) {
      auto th = Clock::now();
      total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
        uint32_t cd = code[i], ct = de9im_transpose(cd);
        bool diag = same && p[i].x == p[i].y;
        A->members(p[i].x, [&](uint32_t ra) { B->members(p[i].y, [&](uint32_t rb) {
          if (!same) { put(f, *A, ra, *B, rb, cd); return; }
          if (distinct) {
            if (ra == rb || (diag && ra > rb)) return;
            if (ra < rb) put(f, *A, ra, *A, rb, cd); else put(f, *A, rb, *A, ra, diag ? cd : ct);
            return;
          }
          put(f, *A, ra, *A, rb, cd);
          if (!diag) put(f, *A, rb, *A, ra, ct);
        }); });
      });
      host_s += secs_since(th);
    });
    cand = st.candidates; hits = st.hits;
  }
  std::string stats = J().k("kind").v("relate").k("pattern").v(pat).k("candidates").u(cand).k("gpu_hits").u(hits).k("rows").u(total)
      .k("host_encode_send_s").n(host_s).k("total_s").n(secs_since(t0)).done();
  if (!c.broken) end_stream(c, stats);
  logf("relate %s x %s: %s%s", a[0].c_str(), a[1].c_str(), stats.c_str(), c.broken ? " (client went away)" : "");
}

// Distance joins in meters (geography semantics): ST_DWithin(a, b, d), or ST_Intersects of the buffered geometries.
static double to_double(const std::string& s) {
  char* e; double v = strtod(s.c_str(), &e);
  if (s.empty() || *e || !std::isfinite(v)) throw std::runtime_error("not a number: " + s);
  return v;
}
static void op_dwithin(Conn& c, const std::vector<std::string>& a) {
  if (a.size() < 6) throw std::runtime_error("DWITHIN needs layer_a, layer_b, mode, dist_a, dist_b, self_pairs[, stats]");
  bool stats_only = a.size() > 6 && a[6] == "stats";
  auto A = get_layer(a[0]), B = get_layer(a[1]);
  if (!A) throw std::runtime_error("no such layer: " + a[0]);
  if (!B) throw std::runtime_error("no such layer: " + a[1]);
  const std::string mode = a[2];
  double da = to_double(a[3]), db = to_double(a[4]);
  bool distinct = a[5] == "distinct";
  if (a[5] != "all" && a[5] != "distinct") throw std::runtime_error("self_pairs must be 'all' or 'distinct'");
  BufferQuery q;
  if (mode == "dwithin") {
    if (da < 0) throw std::runtime_error("distance must be >= 0");
    q.mode = BufferQuery::DWITHIN; q.d = da;
  } else if (mode == "buffer") {
    if (da <= 0 && db <= 0 && (da < 0 || db < 0)) { q.mode = BufferQuery::ERODE; q.ra = -da; q.rb = -db; }
    else if (da >= 0 && db >= 0) { q.mode = BufferQuery::DWITHIN; q.d = da + db; }
    else throw std::runtime_error("buffers of opposite sign are not supported");
  } else throw std::runtime_error("mode must be 'dwithin' or 'buffer'");
  if (A->kind == "point" && B->kind == "point") throw std::runtime_error("point x point distance joins are not supported");
  bool same = A.get() == B.get();
  bool swapped = false;
  if (A->kind == "polygon" && B->kind == "point") { std::swap(A, B); std::swap(q.ra, q.rb); swapped = true; }
  if (A->kind == "point" && q.mode == BufferQuery::ERODE && q.ra > 0) throw std::runtime_error("a point layer cannot have a negative buffer");
  std::lock_guard<std::mutex> g(g_gpu);
  c.u32(GSP_OK); c.streaming = true;
  JoinOpts jo; jo.cancel = &c.broken;
  auto t0 = Clock::now();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  uint64_t total = 0; double host_s = 0;
  auto msize = [](const Layer& L, uint32_t g) -> uint64_t { return L.deduped ? L.moff[g + 1] - L.moff[g] : 1; };
  BufferStats st;
  if (A->kind == "point") {
    const Layer& LA = swapped ? *B : *A; const Layer& LB = swapped ? *A : *B;   // output order (a_id, b_id)
    st = points_buffer_join_stream(A->pts, B->d, B->bvh, q, jo, [&](const uint2* p, size_t n) {
      auto th = Clock::now();
      if (stats_only) { for (size_t i = 0; i < n; i++) total += msize(*B, p[i].y); host_s += secs_since(th); return; }
      total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
        B->members(p[i].y, [&](uint32_t rb) {
          uint32_t ra = p[i].x;
          if (swapped) std::swap(ra, rb);
          f.id(LA.ids.data(ra), LA.ids.len(ra)); f.id(LB.ids.data(rb), LB.ids.len(rb)); f.f64(nan); f.row_done(); });
      });
      host_s += secs_since(th);
    });
  } else {
    jo.upper = same;   // self-join: each unordered pair once, mirrored below
    q.same = same;
    uint8_t* d_need = nullptr;
    if (same && distinct && q.mode == BufferQuery::ERODE) {   // a geometry's pair with itself only matters for groups of identical rows
      std::vector<uint8_t> need(A->n_distinct, 0);
      if (A->deduped) for (uint64_t gi = 0; gi < A->n_distinct; gi++) need[gi] = A->moff[gi + 1] - A->moff[gi] > 1;
      CK(cudaMalloc(&d_need, need.size() + 1)); CK(cudaMemcpy(d_need, need.data(), need.size(), cudaMemcpyHostToDevice));
      q.need_self = d_need;
    }
    try {
      st = buffer_join_stream(A->d, B->d, B->bvh, q, jo, [&](const uint2* p, size_t n) {
        auto th = Clock::now();
        if (stats_only) {
          uint64_t t = 0;
#pragma omp parallel for reduction(+ : t)
          for (int64_t i = 0; i < (int64_t)n; i++) {
            uint64_t ma = msize(*A, p[i].x), mb = msize(*B, p[i].y);
            if (!same) t += ma * mb;
            else if (p[i].x == p[i].y) t += distinct ? ma * (ma - 1) / 2 : ma * ma;
            else t += distinct ? ma * mb : 2 * ma * mb;
          }
          total += t; host_s += secs_since(th); return;
        }
        total += send_parallel(c, n, [&](size_t i, FrameBuf& f) {
          bool diag = same && p[i].x == p[i].y;
          A->members(p[i].x, [&](uint32_t ra) { B->members(p[i].y, [&](uint32_t rb) {
            if (!same) { f.id(A->ids.data(ra), A->ids.len(ra)); f.id(B->ids.data(rb), B->ids.len(rb)); f.f64(nan); f.row_done(); return; }
            if (distinct) {
              if (ra == rb || (diag && ra > rb)) return;
              uint32_t lo = std::min(ra, rb), hi = std::max(ra, rb);
              f.id(A->ids.data(lo), A->ids.len(lo)); f.id(A->ids.data(hi), A->ids.len(hi)); f.f64(nan); f.row_done();
              return;
            }
            f.id(A->ids.data(ra), A->ids.len(ra)); f.id(A->ids.data(rb), A->ids.len(rb)); f.f64(nan); f.row_done();
            if (!diag) { f.id(A->ids.data(rb), A->ids.len(rb)); f.id(A->ids.data(ra), A->ids.len(ra)); f.f64(nan); f.row_done(); }
          }); });
        });
        host_s += secs_since(th);
      });
    } catch (...) { if (d_need) cudaFree(d_need); throw; }
    if (d_need) cudaFree(d_need);
  }
  const char* kind = q.mode == BufferQuery::ERODE ? "buffer_intersects" : "dwithin";
  std::string stats = J().k("kind").v(kind).k("distance_m").n(q.d).k("erode_a_m").n(swapped ? q.rb : q.ra).k("erode_b_m").n(swapped ? q.ra : q.rb)
      .k("candidates").u(st.candidates).k("tested").u(st.pairs).k("gpu_hits").u(st.hits).k("rows").u(total).k("heavy").u(st.heavy)
      .k("traverse_ms").n(st.traverse_ms).k("refine_ms").n(st.refine_ms).k("compact_ms").n(st.compact_ms).k("d2h_ms").n(st.copy_ms)
      .k("host_encode_send_s").n(host_s).k("total_s").n(secs_since(t0)).k("stats_only").b(stats_only).done();
  if (q.mode == BufferQuery::ERODE && A->kind != "point")
    stats = stats.substr(0, stats.size() - 1) + J().k("area_pass_ms").n(st.pass1_ms).k("erode_main_ms").n(st.erode_main_ms)
        .k("erode_tiles_ms").n(st.erode_tiles_ms).k("tiles").u(st.tiles).k("slab_tiles").u(st.slab_tiles).s.replace(0, 1, ",") + "}";
  if (!c.broken) end_stream(c, stats);
  logf("dwithin %s x %s %s %s %s: %s%s", a[0].c_str(), a[1].c_str(), mode.c_str(), a[3].c_str(), a[4].c_str(), stats.c_str(), c.broken ? " (client went away)" : "");
}

static std::string op_status() {
  std::lock_guard<std::mutex> g(g_gpu);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
  std::string ls = "[";
  { std::lock_guard<std::mutex> r(g_reg); bool f = true; for (auto& kv : g_layers) { if (!f) ls += ","; f = false; ls += jstr(kv.first); } }
  return J().k("device").v(p.name).k("cc").v(std::to_string(p.major) + "." + std::to_string(p.minor))
      .k("mem_total_mb").n(tot / 1048576.0).k("mem_free_mb").n(fr / 1048576.0).k("layers").raw(ls + "]").done();
}

static std::string op_list() {
  std::string s = "[";
  std::lock_guard<std::mutex> r(g_reg); bool f = true;
  for (auto& kv : g_layers) { if (!f) s += ","; f = false; s += kv.second->json(); }
  return s + "]";
}

static int to_int(const std::string& s, int dflt) { if (s.empty()) return dflt; char* e; long v = strtol(s.c_str(), &e, 10); if (*e) throw std::runtime_error("not an integer: " + s); return (int)v; }
static bool to_bool(const std::string& s, bool dflt) { if (s.empty()) return dflt; return s == "1" || s == "t" || s == "true" || s == "on"; }

static void handle(int fd) {
  Conn c{fd};
  try {
    uint32_t hdr[3];
    if (!c.read_full(hdr, 12) || hdr[0] != GSP_MAGIC || hdr[2] > 64) { close(fd); return; }
    std::vector<std::string> args(hdr[2]);
    for (auto& s : args) {
      uint32_t l; if (!c.read_full(&l, 4) || l > (64u << 20)) { close(fd); return; }
      s.resize(l); if (l && !c.read_full(&s[0], l)) { close(fd); return; }
    }
    // authenticate: args[0] is the token (constant-time compare), then drop it
    {
      const std::string tok = args.empty() ? std::string() : args[0];
      unsigned diff = tok.size() != g_token.size();
      for (size_t i = 0; i < g_token.size(); i++) diff |= (unsigned)(uint8_t)g_token[i] ^ (unsigned)(uint8_t)(i < tok.size() ? tok[i] : 0);
      if (diff) { logf("rejected request: bad token"); c.error("authentication failed"); close(fd); return; }
      if (!args.empty()) args.erase(args.begin());
    }
    switch (hdr[1]) {
      case GSP_OP_LOAD: {
        if (args.size() < 7) throw std::runtime_error("LOAD needs 7 args");
        LoadSpec sp; sp.table = args[1]; sp.id_col = args[2]; sp.geom_col = args[3]; sp.filter = args[4];
        c.ok_json(build_layer(args[0], sp, to_int(args[5], 0), to_bool(args[6], true), "table " + args[1] + (args[4].empty() ? "" : " where " + args[4])));
        break;
      }
      case GSP_OP_LOAD_QUERY: {
        if (args.size() < 4) throw std::runtime_error("LOAD_QUERY needs 4 args");
        LoadSpec sp; sp.sql = args[1];
        c.ok_json(build_layer(args[0], sp, to_int(args[2], 0), to_bool(args[3], true), "query"));
        break;
      }
      case GSP_OP_DROP: {
        if (args.size() < 1) throw std::runtime_error("DROP needs layer");
        std::shared_ptr<Layer> old;
        { std::lock_guard<std::mutex> g(g_reg); auto it = g_layers.find(args[0]); if (it != g_layers.end()) { old = it->second; g_layers.erase(it); } }
        c.ok_json(old ? "true" : "false");
        break;
      }
      case GSP_OP_LIST: c.ok_json(op_list()); break;
      case GSP_OP_STATUS: c.ok_json(op_status()); break;
      case GSP_OP_JOIN: op_join(c, args); break;
      case GSP_OP_CLUSTER: op_cluster(c, args); break;
      case GSP_OP_RELATE: op_relate(c, args); break;
      case GSP_OP_DWITHIN: op_dwithin(c, args); break;
      default: c.error("unknown op");
    }
  } catch (const std::exception& e) {
    logf("error: %s", e.what());
    if (c.streaming) { c.u32(GSP_FRAME_ERR); c.str(e.what()); } else c.error(e.what());
  }
  close(fd);
}

// ------------------------------------------------------------------------------------------------- main
static int listen_unix(const std::string& path) {
  int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  sockaddr_un a{}; a.sun_family = AF_UNIX; strncpy(a.sun_path, path.c_str(), sizeof(a.sun_path) - 1);
  unlink(path.c_str());
  if (bind(fd, (sockaddr*)&a, sizeof a) || listen(fd, 64)) { perror(("unix socket " + path).c_str()); exit(1); }
  chmod(path.c_str(), 0666);
  return fd;
}
static int listen_tcp(const std::string& hp) {
  auto pos = hp.rfind(':');
  std::string host = hp.substr(0, pos); int port = atoi(hp.c_str() + pos + 1);
  int fd = socket(AF_INET, SOCK_STREAM, 0), one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) { fprintf(stderr, "bad tcp address %s\n", hp.c_str()); exit(1); }
  if (bind(fd, (sockaddr*)&a, sizeof a) || listen(fd, 64)) { perror(("tcp " + hp).c_str()); exit(1); }
  return fd;
}

int main(int argc, char** argv) {
  std::string sock = "/tmp/gpujoind.sock", tcp;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--conninfo" && i + 1 < argc) g_conninfo = argv[++i];
    else if (a == "--socket" && i + 1 < argc) sock = argv[++i];
    else if (a == "--tcp" && i + 1 < argc) tcp = argv[++i];
    else if (a == "--threads" && i + 1 < argc) g_threads = atoi(argv[++i]);
    else if (a == "--token-file" && i + 1 < argc) {
      std::ifstream tf(argv[++i]); std::getline(tf, g_token);
      if (g_token.empty()) { fprintf(stderr, "empty token file\n"); return 1; }
    }
    else { fprintf(stderr, "usage: gpujoind --conninfo CONNINFO [--socket PATH] [--tcp HOST:PORT] [--threads N] [--token-file FILE]\n"); return 1; }
  }
  if (g_conninfo.empty()) { fprintf(stderr, "--conninfo is required\n"); return 1; }
  if (!tcp.empty() && g_token.empty()) { fprintf(stderr, "--tcp requires --token-file (the daemon runs SQL with its own credentials)\n"); return 1; }
  signal(SIGPIPE, SIG_IGN);
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); CK(cudaFree(0));
  size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
  logf("gpujoind on %s (cc %d.%d), %.1f GB free of %.1f GB", p.name, p.major, p.minor, fr / 1e9, tot / 1e9);
  std::vector<pollfd> fds;
  fds.push_back({listen_unix(sock), POLLIN, 0}); logf("listening on unix:%s", sock.c_str());
  if (!tcp.empty()) { fds.push_back({listen_tcp(tcp), POLLIN, 0}); logf("listening on tcp:%s", tcp.c_str()); }
  for (;;) {
    if (poll(fds.data(), fds.size(), -1) < 0) { if (errno == EINTR) continue; perror("poll"); return 1; }
    for (auto& f : fds) if (f.revents & POLLIN) {
      int cfd = accept(f.fd, nullptr, nullptr);
      if (cfd < 0) continue;
      int one = 1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
      std::thread(handle, cfd).detach();
    }
  }
}
