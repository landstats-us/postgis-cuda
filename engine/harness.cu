// Test harness: join two .gset files on the GPU, write intersecting pairs (+area) to CSV.
//   harness P.gset|P.pset R.gset [--area] [--out f.csv] [--reps N] [--subdiv N | --subdiv-a N --subdiv-b N] [--dedup] [--cluster]
//           [--dwithin D | --erode R[,RB]]   (meters: ST_DWithin(geography) / ST_Intersects of ST_Buffer(geography, -R))
// All geometry is FP64.
#include "engine.cuh"
#include "buffer.cuh"
#include "subdivide.h"
#include "dedup.h"
#include <chrono>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// One side of the join as prepared for the GPU: optional dedup, then optional subdivision.
struct Side {
  const HostSet* base = nullptr;   // geometries the GPU sees as "parents"
  Dedup dd; bool deduped = false;
  HostSet pieces; std::vector<uint32_t> parent; std::vector<uint8_t> syn; bool subdivided = false;
  void prepare(const HostSet& h, bool dedup, int subdiv) {
    base = &h;
    if (dedup) { dd = dedup_set(h); deduped = true; base = &dd.u; }
    if (subdiv > 0) { pieces = subdivide_set(*base, subdiv, parent, syn); subdivided = true; }
  }
  DSet upload() const { return subdivided ? upload_set(pieces, &parent, &syn) : upload_set(*base); }
  uint64_t gpu_n() const { return subdivided ? pieces.n : base->n; }
  // members of a base geometry in the original set
  template <class F> void members(uint32_t g, F&& f) const {
    if (!deduped) { f(g); return; }
    for (uint64_t k = dd.moff[g]; k < dd.moff[g + 1]; k++) f(dd.midx[k]);
  }
};

static void run(const HostSet& hp, const HostSet& hr, bool same, bool area, const char* out_csv, int reps,
                int sub_a, int sub_b, bool dedup, bool relate) {
  auto t0 = Clock::now();
  Side A, B;
  A.prepare(hp, dedup, sub_a);
  const Side* pB = &A;
  if (!same || sub_a != sub_b) { B.prepare(hr, dedup, sub_b); pB = &B; }
  double prep_ms = ms_since(t0);
  printf("prepare: P %llu geoms -> %llu distinct -> %llu gpu geoms; R %llu -> %llu -> %llu (%.0f ms host)\n",
         (unsigned long long)hp.n, (unsigned long long)A.base->n, (unsigned long long)A.gpu_n(),
         (unsigned long long)hr.n, (unsigned long long)pB->base->n, (unsigned long long)pB->gpu_n(), prep_ms);
  t0 = Clock::now();
  DSet P = A.upload(), R = pB->upload();
  CK(cudaDeviceSynchronize());
  double up_ms = ms_since(t0);
  GpuTimer tm; tm.start(); Bvh bvh = build_bvh(R); float bvh_ms = tm.stop();
  RefineParams rp; rp.want_area = area ? 1 : 0; rp.want_relate = relate ? 1 : 0;
  JoinOpts jo;
  if (getenv("BATCH_Q")) jo.batch_q = atoi(getenv("BATCH_Q"));
  if (getenv("MAX_PAIRS")) jo.max_pairs = atoll(getenv("MAX_PAIRS"));
  if (getenv("NO_AGG")) jo.no_agg = true;
  bool self = same && sub_a == sub_b && !getenv("NO_UPPER");
  jo.upper = self;
  std::vector<uint2> pairs; std::vector<double> areas; std::vector<uint32_t> codes; JoinStats st;
  for (int i = 0; i < reps; i++) {
    pairs.clear(); areas.clear(); codes.clear();
    st = join_stream(P, R, bvh, rp, jo, [&](const uint2* pp, const double* aa, const uint32_t* cc, size_t n) {
      pairs.insert(pairs.end(), pp, pp + n); if (aa) areas.insert(areas.end(), aa, aa + n); if (cc) codes.insert(codes.end(), cc, cc + n); });
  }
  if (jo.no_agg) { for (size_t i = 0; i < pairs.size(); i++) printf("piece pair a=%u b=%u area=%.9g\n", pairs[i].x, pairs[i].y, area ? areas[i] : 0.0); return; }
  // expand distinct-geometry results to original rows
  auto te = Clock::now();
  std::vector<uint2> rows; std::vector<double> rarea; std::vector<uint32_t> rcode;
  if (A.deduped || pB->deduped || self) {
    for (size_t i = 0; i < pairs.size(); i++)
      A.members(pairs[i].x, [&](uint32_t a) { pB->members(pairs[i].y, [&](uint32_t b) {
        rows.push_back(make_uint2(a, b)); if (area) rarea.push_back(areas[i]); if (relate) rcode.push_back(codes[i]);
        if (self && pairs[i].x != pairs[i].y) {   // mirrored row: relate(b, a) is the transpose
          rows.push_back(make_uint2(b, a)); if (area) rarea.push_back(areas[i]); if (relate) rcode.push_back(de9im_transpose(codes[i])); } }); });
  } else { rows.swap(pairs); rarea.swap(areas); rcode.swap(codes); }
  double exp_ms = ms_since(te);
  double asum = 0; for (double a : rarea) asum += a;
  printf("P=%u R=%u store=%.1fMB | upload=%.0fms bvh=%.1fms traverse=%.2fms refine=%.2fms compact=%.2fms d2h=%.2fms expand(host)=%.1fms | cand=%llu gpu_hits=%llu rows=%zu area_sum=%.9g heavy=%llu\n",
         P.n, R.n, (P.bytes + R.bytes) / 1e6, up_ms, bvh_ms, st.traverse_ms, st.refine_ms, st.compact_ms, st.copy_ms, exp_ms,
         (unsigned long long)st.candidates, (unsigned long long)st.hits, rows.size(), asum, (unsigned long long)st.heavy);
  if (out_csv) {
    FILE* f = fopen(out_csv, "w"); fprintf(f, relate ? "landrecid,sid,isect,relate\n" : "landrecid,sid,isect,area\n");
    char m[10] = {0};
    for (size_t i = 0; i < rows.size(); i++) {
      if (relate) { de9im_str(rcode[i], m); fprintf(f, "%s,\"%s\",t,%s\n", hp.ids[rows[i].x].c_str(), hr.ids[rows[i].y].c_str(), m); }
      else fprintf(f, "%s,\"%s\",t,%.17g\n", hp.ids[rows[i].x].c_str(), hr.ids[rows[i].y].c_str(), area ? rarea[i] : 0.0);
    }
    fclose(f);
  }
  free_set(P); free_set(R); free_bvh(bvh);
}

static void run_buffer(const HostSet& hp, const HostSet& hr, bool same, const BufferQuery& bq, const char* out_csv, int reps,
                       int sub_a, int sub_b, bool dedup) {
  auto t0 = Clock::now();
  Side A, B;
  A.prepare(hp, dedup, sub_a);
  const Side* pB = &A;
  if (!same || sub_a != sub_b) { B.prepare(hr, dedup, sub_b); pB = &B; }
  printf("prepare: P %llu -> %llu gpu geoms; R %llu -> %llu (%.0f ms host)\n", (unsigned long long)hp.n, (unsigned long long)A.gpu_n(),
         (unsigned long long)hr.n, (unsigned long long)pB->gpu_n(), ms_since(t0));
  DSet P = A.upload(), R = pB->upload();
  CK(cudaDeviceSynchronize());
  Bvh bvh = build_bvh(R);
  JoinOpts jo;
  if (getenv("BATCH_Q")) jo.batch_q = atoi(getenv("BATCH_Q"));
  if (getenv("MAX_PAIRS")) jo.max_pairs = atoll(getenv("MAX_PAIRS"));
  bool self = same && sub_a == sub_b && !getenv("NO_UPPER");
  jo.upper = self;
  // self-join: a geometry's pair with itself matters only for dedup groups (identical geometries) of 2+ rows
  BufferQuery q = bq; q.same = self;
  uint8_t* d_need = nullptr;
  if (self) {
    std::vector<uint8_t> need(A.base->n, 0);
    if (A.deduped) for (uint64_t g = 0; g < A.base->n; g++) need[g] = A.dd.moff[g + 1] - A.dd.moff[g] > 1;
    CK(cudaMalloc(&d_need, need.size() + 1)); CK(cudaMemcpy(d_need, need.data(), need.size(), cudaMemcpyHostToDevice));
    q.need_self = d_need;
  }
  std::vector<uint2> pairs; BufferStats st;
  for (int i = 0; i < reps; i++) {
    pairs.clear();
    st = buffer_join_stream(P, R, bvh, q, jo, [&](const uint2* pp, size_t n) { pairs.insert(pairs.end(), pp, pp + n); });
  }
  std::vector<uint2> rows;
  for (size_t i = 0; i < pairs.size(); i++)
    A.members(pairs[i].x, [&](uint32_t a) { pB->members(pairs[i].y, [&](uint32_t b) {
      rows.push_back(make_uint2(a, b));
      if (self && pairs[i].x != pairs[i].y) rows.push_back(make_uint2(b, a)); }); });
  printf("P=%u R=%u | traverse=%.2fms refine=%.2fms compact=%.2fms d2h=%.2fms | cand=%llu pairs=%llu heavy=%llu gpu_hits=%llu rows=%zu\n",
         P.n, R.n, st.traverse_ms, st.refine_ms, st.compact_ms, st.copy_ms, (unsigned long long)st.candidates, (unsigned long long)st.pairs,
         (unsigned long long)st.heavy, (unsigned long long)st.hits, rows.size());
  if (out_csv) {
    FILE* f = fopen(out_csv, "w"); fprintf(f, "landrecid,sid,isect,area\n");
    for (auto& r : rows) fprintf(f, "%s,\"%s\",t,0\n", hp.ids[r.x].c_str(), hr.ids[r.y].c_str());
    fclose(f);
  }
  free_set(P); free_set(R); free_bvh(bvh); if (d_need) cudaFree(d_need);
}

static void run_points_buffer(const HostPoints& pts, const HostSet& hr, const BufferQuery& bq, const char* out_csv, int reps, int sub_b, bool dedup) {
  Side B; B.prepare(hr, dedup, sub_b);
  PSetD P = upload_points(pts.xy); DSet R = B.upload(); CK(cudaDeviceSynchronize());
  Bvh bvh = build_bvh(R);
  JoinOpts jo; std::vector<uint2> pairs; BufferStats st;
  for (int i = 0; i < reps; i++) {
    pairs.clear();
    st = points_buffer_join_stream(P, R, bvh, bq, jo, [&](const uint2* pp, size_t n) { pairs.insert(pairs.end(), pp, pp + n); });
  }
  std::vector<uint2> rows;
  for (size_t i = 0; i < pairs.size(); i++) B.members(pairs[i].y, [&](uint32_t b) { rows.push_back(make_uint2(pairs[i].x, b)); });
  printf("points=%u R=%u | traverse=%.2fms refine=%.2fms compact=%.2fms d2h=%.2fms | cand=%llu pairs=%llu gpu_hits=%llu rows=%zu\n",
         P.n, R.n, st.traverse_ms, st.refine_ms, st.compact_ms, st.copy_ms, (unsigned long long)st.candidates, (unsigned long long)st.pairs,
         (unsigned long long)st.hits, rows.size());
  if (out_csv) {
    FILE* f = fopen(out_csv, "w"); fprintf(f, "landrecid,sid,isect,area\n");
    for (auto& r : rows) fprintf(f, "%s,\"%s\",t,0\n", pts.ids[r.x].c_str(), hr.ids[r.y].c_str());
    fclose(f);
  }
  free_points(P); free_set(R); free_bvh(bvh);
}

static void run_points(const HostPoints& pts, const HostSet& hr, const char* out_csv, int reps, int sub_b, bool dedup) {
  auto t0 = Clock::now();
  Side B; B.prepare(hr, dedup, sub_b);
  printf("prepare: points %llu; R %llu -> %llu distinct -> %llu gpu geoms (%.0f ms host)\n", (unsigned long long)pts.n,
         (unsigned long long)hr.n, (unsigned long long)B.base->n, (unsigned long long)B.gpu_n(), ms_since(t0));
  t0 = Clock::now();
  PSetD P = upload_points(pts.xy); DSet R = B.upload(); CK(cudaDeviceSynchronize());
  double up_ms = ms_since(t0);
  GpuTimer tm; tm.start(); Bvh bvh = build_bvh(R); float bvh_ms = tm.stop();
  JoinOpts jo; std::vector<uint2> pairs; std::vector<uint8_t> where; PointJoinStats st;
  for (int i = 0; i < reps; i++) {
    pairs.clear(); where.clear();
    st = join_points_stream(P, R, bvh, jo, [&](const uint2* pp, const uint8_t* ww, size_t n) {
      pairs.insert(pairs.end(), pp, pp + n); where.insert(where.end(), ww, ww + n); });
  }
  auto te = Clock::now();
  std::vector<uint2> rows; std::vector<uint8_t> rwhere;
  for (size_t i = 0; i < pairs.size(); i++) B.members(pairs[i].y, [&](uint32_t b) { rows.push_back(make_uint2(pairs[i].x, b)); rwhere.push_back(where[i]); });
  printf("points=%u R=%u | upload=%.0fms bvh=%.1fms traverse=%.2fms refine=%.2fms compact=%.2fms d2h=%.2fms expand(host)=%.1fms | cand=%llu gpu_hits=%llu rows=%zu\n",
         P.n, R.n, up_ms, bvh_ms, st.traverse_ms, st.refine_ms, st.compact_ms, st.copy_ms, ms_since(te),
         (unsigned long long)st.candidates, (unsigned long long)st.hits, rows.size());
  if (out_csv) {
    FILE* f = fopen(out_csv, "w"); fprintf(f, "landrecid,sid,isect,relate\n");
    char m[10] = {0};
    for (size_t i = 0; i < rows.size(); i++) {
      de9im_str(rwhere[i] == 2 ? DE9IM_PT_BOUNDARY : DE9IM_PT_INTERIOR, m);
      fprintf(f, "%s,\"%s\",t,%s\n", pts.ids[rows[i].x].c_str(), hr.ids[rows[i].y].c_str(), m);
    }
    fclose(f);
  }
  free_points(P); free_set(R); free_bvh(bvh);
}

static void run_cluster(const HostSet& hp, const char* out_csv, int reps, int sub, bool dedup) {
  auto t0 = Clock::now();
  Side A; A.prepare(hp, dedup, sub);
  printf("prepare: %llu geoms -> %llu distinct -> %llu gpu geoms (%.0f ms host)\n", (unsigned long long)hp.n,
         (unsigned long long)A.base->n, (unsigned long long)A.gpu_n(), ms_since(t0));
  t0 = Clock::now(); DSet P = A.upload(); CK(cudaDeviceSynchronize()); double up_ms = ms_since(t0);
  GpuTimer tm; tm.start(); Bvh bvh = build_bvh(P); float bvh_ms = tm.stop();
  RefineParams rp; rp.want_area = 0;
  std::vector<uint32_t> root; ClusterStats cs;
  for (int i = 0; i < reps; i++) root = cluster_intersecting(P, (uint32_t)A.base->n, bvh, rp, JoinOpts(), &cs);
  // label every original row; dense cluster numbers in order of first row
  auto te = Clock::now();
  std::vector<uint32_t> label(hp.n), dense(root.size(), UINT32_MAX), row_root(hp.n); uint32_t k = 0;
  for (uint32_t g = 0; g < A.base->n; g++) A.members(g, [&](uint32_t r) { row_root[r] = root[g]; });
  for (uint64_t r = 0; r < hp.n; r++) { uint32_t rt = row_root[r]; if (dense[rt] == UINT32_MAX) dense[rt] = k++; label[r] = dense[rt]; }
  printf("cluster: n=%llu | upload=%.0fms bvh=%.1fms traverse=%.2fms refine=%.2fms compact=%.2fms union-find=%.2fms label(host)=%.1fms | cand=%llu edges=%llu clusters=%u\n",
         (unsigned long long)hp.n, up_ms, bvh_ms, cs.join.traverse_ms, cs.join.refine_ms, cs.join.compact_ms, cs.uf_ms, ms_since(te),
         (unsigned long long)cs.join.candidates, (unsigned long long)cs.join.hits, k);
  if (out_csv) { FILE* f = fopen(out_csv, "w"); fprintf(f, "id,cluster\n"); for (uint64_t r = 0; r < hp.n; r++) fprintf(f, "%s,%u\n", hp.ids[r].c_str(), label[r]); fclose(f); }
  free_set(P); free_bvh(bvh);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: harness P.gset|P.pset R.gset [--area] [--out f.csv] [--reps N] "
                    "[--subdiv N | --subdiv-a N --subdiv-b N] [--dedup] [--cluster] [--relate]\n");
    return 1;
  }
  bool area = false, dedup = false, cluster = false, relate = false; const char* out = nullptr; int reps = 3;
  BufferQuery bq; bool buffered = false;
  int sub_a = 0, sub_b = 0;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--area")) area = true;
    else if (!strcmp(argv[i], "--dedup")) dedup = true;
    else if (!strcmp(argv[i], "--cluster")) cluster = true;
    else if (!strcmp(argv[i], "--relate")) relate = true;
    else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--dwithin") && i + 1 < argc) { bq.mode = BufferQuery::DWITHIN; bq.d = atof(argv[++i]); buffered = true; }
    else if (!strcmp(argv[i], "--erode") && i + 1 < argc) {
      bq.mode = BufferQuery::ERODE; const char* v = argv[++i]; bq.ra = bq.rb = atof(v);
      if (const char* c = strchr(v, ',')) bq.rb = atof(c + 1);
      buffered = true;
    }
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--subdiv") && i + 1 < argc) sub_a = sub_b = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--subdiv-a") && i + 1 < argc) sub_a = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--subdiv-b") && i + 1 < argc) sub_b = atoi(argv[++i]);
    else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
  }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); printf("device: %s (cc %d.%d)\n", pr.name, pr.major, pr.minor);
  if (strstr(argv[1], ".pset")) {   // points x polygons
    HostPoints pts = load_pset(argv[1]); HostSet hr = load_gset(argv[2]);
    if (buffered) run_points_buffer(pts, hr, bq, out, reps, sub_b, dedup);
    else run_points(pts, hr, out, reps, sub_b, dedup);
    return 0;
  }
  bool same = !strcmp(argv[1], argv[2]);
  HostSet hp = load_gset(argv[1]);
  if (cluster) { run_cluster(hp, out, reps, sub_a, dedup); return 0; }   // ST_ClusterIntersecting over P (R ignored)
  HostSet hr_own; if (!same) hr_own = load_gset(argv[2]);
  const HostSet& hr = same ? hp : hr_own;
  if (buffered) run_buffer(hp, hr, same, bq, out, reps, sub_a, sub_b, dedup);
  else run(hp, hr, same, area, out, reps, sub_a, sub_b, dedup, relate);
  return 0;
}
