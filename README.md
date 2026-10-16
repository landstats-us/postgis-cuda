# postgis-cuda: GPU spatial joins for PostgreSQL/PostGIS (CUDA, Hopper / sm_90)

GPU-resident spatial joins for PostGIS data: `ST_Intersects`, `ST_Area(ST_Intersection(a, b))`, `ST_Relate`
(DE-9IM), point-in-polygon, `ST_ClusterIntersecting`, `ST_DWithin(geography)` and
`ST_Intersects(ST_Buffer(a::geography, -r), ST_Buffer(b::geography, -r))`, validated pair-for-pair against
PostGIS 3.6 on parcel data. All geometry is FP64.

```
 Postgres backend (gpu_spatial.so)  --unix socket / tcp-->  gpujoind (CUDA, owns one GPU)
   gpu_load(...)            ---------------------------->  parallel COPY (binary) from Postgres
   gpu_join(a, b)           <---- streamed row frames ----  layers resident in GPU memory
   gpu_cluster_intersecting                                 LBVH + exact refinement kernels
```

* `engine/`  CUDA engine (header-only): device layer format, LBVH, refinement kernels, subdivision,
  deduplication, union-find clustering, plus `harness.cu` (file-based test/benchmark driver).
* `daemon/`  `gpujoind`: layer registry, libpq loader, socket server. `protocol.h` is the wire format.
* `ext/`     `gpu_spatial` Postgres 18 extension (thin C client, no CUDA, no PostGIS linkage).
* `scripts/` toolchain setup, data export, PostGIS ground truth, validation, benchmarks.
* `tools/gsp_cat.c` raw protocol client (throughput tests).

## Algorithms

* **Candidates**: LBVH (Karras 2012) over float bboxes rounded outward; one thread per probe geometry,
  two-pass count/scan/write, batched so candidate buffers stay bounded.
* **Refinement** (one warp per candidate pair): edges of both polygons that touch the pair's overlap box are
  collected into shared memory; boundary contact by exact segment tests; containment by crossing parity
  from a reference point; area of intersection by Green's theorem over the parts of each boundary inside the
  other polygon (no intersection polygon is materialized). Coincident edges use a net-orientation rule.
  Pairs whose local edge lists exceed shared memory go to a multi-warp path (32-edge tiles across the GPU).
* **Exactness** (FP64): coordinates are stored as `v - origin` (exact by Sterbenz), so shared vertices of
  neighbouring parcels stay bit-identical in every pair frame; orientation uses a compensated (FMA)
  determinant; points exactly on an edge are classified by simulation of simplicity so a point on a shared
  boundary falls in exactly one side. Edge pairs are coincident if either edge lies within 1e-11 deg of the
  other's line (decided identically from both polygons). A boundary piece lying on the other polygon's
  boundary is weighted by the product rule jump(1_A 1_B) = jump(1_A) avg(1_B) + jump(1_B) avg(1_A), i.e. 1/2
  where the other polygon covers exactly one side, which is exact and symmetric for shared boundaries,
  identical (stacked) parcels and zero-width cut edges.
* **Self-joins** test each unordered pair once (a <= b) and mirror rows when both orders are requested
  (relate(b, a) is the transposed matrix).
* **ST_Relate**: the edge pass also measures, per pair, the length of each polygon's real boundary strictly
  inside the other and lying on its boundary, plus same/opposite-side coincidence and point-contact flags.
  "Outside" = total boundary length - inside - on, which stays correct when either polygon is subdivided
  (inside/on sum across pieces; outside does not). The nine DE-9IM cells follow from these (e.g. II = 2 iff a
  boundary enters the other interior or the boundaries coincide with interiors on the same side; EE = 2).
  Coincidence is exact (orientation sign), except at subdivision cut points (rounded, <= 1e-11 deg).
* **Subdivision** (layer option, like `ST_Subdivide`): polygons with more than N vertices are cut into pieces;
  edges created by cuts are flagged and ignored for boundary contact; piece results are folded back per row
  (areas summed, intersects OR-ed) on the GPU.
* **Deduplication** (layer option): identical geometries (stacked parcels) are stored once; results expand to
  all member rows. 59% of Johnson County's intersecting parcel pairs and 10.8M of LA's are such duplicates.
* **ST_ClusterIntersecting**: intersects edges are fed batch by batch into a lock-free union-find on the GPU.
* **ST_DWithin (meters, geography)**: query boxes are widened by d (in degrees, conservatively for the box's
  latitudes); per candidate piece pair, real boundary edges are compared in a local WGS84 metric frame
  (meridional / prime-vertical radii at the window's latitude), plus containment by parity. Edge pairs whose
  distance lies within a narrow band around d (frame error + the up-to L^2 tan(lat)/8R gap between straight
  lon/lat edges and great-circle arcs) are decided as PostGIS does: closest points of the great-circle edges on
  the sphere (ported `edge_distance_to_point`, robust cross product), then the WGS84 geodesic (Vincenty).
* **Negative buffers** (`ST_Intersects(ST_Buffer(a::geography, -ra), ST_Buffer(b::geography, -rb))`), without
  building buffers. The eroded polygon X_A is A minus the points within r of its boundary, with GEOS's round
  joins as GEOS draws them (an inscribed fillet of round(angle / (pi/16)) chords at each reflex vertex, i.e.
  8 segments per quarter circle). Its boundary lies on the "elements": every edge offset inward by r and the
  fillet chords. X_A and X_B intersect iff an A element crosses a B element at a point outside both
  exclusion zones (two-sided strip per edge, fillet per reflex vertex, exterior disk per convex vertex), or a
  vertex of one eroded boundary (element endpoint or element-element crossing outside its own zone) lies in
  the other eroded polygon. Only the window shrink(bbox A, ra) ∩ shrink(bbox B, rb) is examined (candidates
  whose window is empty are dropped before anything else). Pairs are
  evaluated per parent geometry (all pieces at once: the boundary that erodes a piece may belong to a
  neighbour). Geography's buffer runs in the UTM zone `_ST_BestSRID` picks, where r projected meters are r/k
  true meters; k is applied per side. Two passes: the exact intersects + area join first, then only pairs
  whose overlap could hold a disk of radius min(ra, rb) (area >= pi (0.98 r)^2) are eroded; 1,645 of 167,571
  intersecting Johnson Co. parent pairs, 183,136 of 6.24M in LA. Pairs whose local edge lists overflow shared
  memory are split into window tiles (quadrants, recursively) until each tile fits; every vertex of the
  eroded intersection lies in some tile.

## Results (H200 NVL, GPU 3; PostGIS 3.6.3 on the production server)

GPU time = candidate search + refinement + compaction with layers resident (excludes one-time load and
row delivery). Every row below matches PostGIS exactly (0 missing, 0 extra pairs; identical cluster partition).

| Workload | Rows | PostGIS | GPU | Speedup |
|---|---|---|---|---|
| Johnson Co. parcels x soils (area) | 237,231 | 1.45 s (48 conns) | 4.2 ms fp32 / 8.8 ms fp64 | ~170-350x |
| Johnson Co. parcels x parcels (area) | 1,185,533 | 2.16 s (48 conns) | 42 ms fp64 | ~50x |
| Johnson Co. parcels x counties (area) | 74,258 | 0.27 s (48 conns) | 0.5 ms | ~540x |
| Johnson Co. ST_ClusterIntersecting | 2,105 clusters | 1.83 s (1 core) | 35 ms | ~52x |
| LA parcels x soils (area) | 2,743,450 | 18.0 s (48 conns) | 21 ms fp32 | ~860x |
| LA parcels x counties (area) | 2,419,854 | 17.7 s (48 conns) | 7.1 ms fp32 / 7.7 ms fp64 | ~2,300x |
| LA parcels x parcels (area) | 23,232,840 | 61.5 s (48 conns) | 297 ms fp64 | ~207x |
| LA addresses x parcels (3.97M points) | 14,078,923 | 13.3 s (48 conns) | 6.5 ms | ~2,000x |
| LA ST_ClusterIntersecting | 81,629 clusters | 32.5 s (1 core) | 133 ms (no subdivision) | ~245x |

With parcels subdivided at 256 vertices and self-joins evaluated once per unordered pair: Johnson Co.
parcels x parcels 6 ms, clustering 1.9 ms; LA parcels x parcels 139 ms.

Distance and buffer joins (meters, geography; subdivided and deduplicated layers, self-joins once per pair):

| Workload | Rows | PostGIS | GPU |
|---|---|---|---|
| Johnson Co. parcels x parcels, `ST_DWithin(geog, geog, 5)` | 1,197,956 | 76.2 s (48 conns) | 1.9 ms |
| Johnson Co. parcels x counties, `ST_DWithin(geog, geog, 1000)` | 76,158 | 0.41 s (48 conns) | 0.6 ms |
| Johnson Co. parcels x parcels, `ST_Intersects(ST_Buffer(geog, -1), ST_Buffer(geog, -1))` | 708,951 | 6.6 s (1 conn: buffers 0.27 s, index 0.08 s, join 6.3 s) | 13 ms |
| LA parcels x parcels, `ST_DWithin(geog, geog, 5)` | 23,800,370 | 79.3 s (48 conns) | 150 ms |
| LA addresses x parcels, `ST_DWithin(geog, geog, 5)` | 14,992,942 | 21.0 s (48 conns) | 9 ms |
| LA parcels x parcels, buffer(-1) x buffer(-1) | 10,913,792 | 14.8 s (48 conns: buffers 3.9 s, index 3.4 s, join 7.5 s) | 300 ms |
| LA addresses in buffer(parcel, -1) | 13,914,022 | 12.1 s (48 conns, buffers prebuilt) | 9 ms |

Full table (all 162M parcels, 155M distinct geometries; 161M address points; resident layers 70 GB; stats mode,
i.e. without row delivery):

| Workload | Rows | GPU | Peak scratch |
|---|---|---|---|
| parcels x parcels, `ST_DWithin(geog, geog, 5)` | 1,313,097,219 | 13.2 s | 6 GiB |
| parcels x parcels, buffer(-1) x buffer(-1) | 608,826,258 | 46.7 s (area pass 30.2 s, erosion 4.0 s + tiles 6.0 s) | 30 GiB |
| parcels x parcels, buffer(-4) x buffer(-4) | 586,394,839 | 51.4 s | 30 GiB |
| addresses x parcels, `ST_DWithin(geog, geog, 5)` | 636,569,040 | 3.3 s | 4 GiB |
| addresses in buffer(parcel, -1) | 593,818,008 | 2.7 s | 4 GiB |
| parcels x counties, `ST_DWithin(geog, geog, 1000)` | 176,479,226 | 2.6 s | 3 GiB |

For scale, the plain intersects self-join is 10.5 s. The buffer joins are bounded by their first pass (the
exact intersects + area join, ~30 s), which PostGIS avoids by building every buffer once and indexing them;
extrapolating the LA timings, PostGIS needs ~16 min on 48 cores for the full buffer(-1) self-join and ~70 min
for the 5 m dwithin.

All match PostGIS exactly except LA parcels x parcels dwithin, where PostGIS returns 21 extra pairs that are
7-12 m apart: `ST_DWithin(geography)` reports them inside that join (also for a single outer row) but not
when called on the pair alone, and `_ST_DWithinUncached` and the UTM distance agree with the GPU (a PostGIS
cached-path bug). The LA buffer join spends ~150 ms in pass 1 (intersects + area, after dropping candidates
whose shrunk bboxes cannot overlap) and ~35 ms eroding the 121k pairs that remain.

Through SQL (dev instance, both counties loaded, end to end including row delivery into Postgres):
parcels x counties 2.49M rows 0.46 s; parcels x parcels 24.4M rows 4.6 s; addresses x parcels 14.1M rows
2.0 s; clustering 2.49M rows 0.86 s. Large result sets are bounded by Postgres' set-returning-function
path (~5-6M rows/s per backend), not the GPU.

### Including the one-time load

The GPU columns above exclude loading: a layer is loaded once from Postgres (parallel binary COPY, dedup,
subdivision, upload) and then serves any number of queries. `scripts/bench_amortized.py SOCKET` measures both from
scratch against the production database: every dataset loaded straight from its table into the daemon, then every
workload run on those layers through the daemon (stats mode: rows counted, not delivered to SQL).

| Layer | Rows | Load | (COPY / prep / upload) |
|---|---|---|---|
| Johnson Co. parcels (`combine.parcel_enriched_19103`, subdivide 256, dedup) | 74,133 | 0.37 s | 0.22 / 0.11 / 0.04 s |
| LA parcels (`combine.parcel_enriched_06037`) | 2,418,914 | 2.38 s | 1.16 / 0.89 / 0.24 s |
| counties (`tiger_data.county_4326`, subdivide 64) | 3,235 | 0.47 s | 0.18 / 0.26 / 0.02 s |
| Johnson Co. soils (SSURGO, bbox query) | 34,023 | 1.54 s | 1.04 / 0.41 / 0.05 s |
| LA soils | 21,575 | 1.23 s | 0.81 / 0.37 / 0.05 s |
| LA addresses (bbox query on the 161M-row table) | 3,973,459 | 7.62 s | 7.49 / 0 / 0.12 s |

| Workload | Rows | Load | Query | Load + query | PostGIS |
|---|---|---|---|---|---|
| Johnson Co. parcels x soils (area) | 237,231 | 1.91 s | 52 ms | 1.96 s | 1.45 s (48 conns) |
| Johnson Co. parcels x parcels (area) | 1,185,533 | 0.37 s | 72 ms | 0.44 s | 2.16 s (48 conns) |
| Johnson Co. parcels x counties (area) | 74,258 | 0.84 s | 61 ms | 0.90 s | 0.27 s (48 conns) |
| Johnson Co. ST_ClusterIntersecting | 74,133 | 0.37 s | 69 ms | 0.44 s | 1.83 s (1 core) |
| Johnson Co. parcels x parcels, dwithin 5 m | 1,197,956 | 0.37 s | 54 ms | 0.42 s | 76.2 s (48 conns) |
| Johnson Co. parcels x counties, dwithin 1 km | 76,158 | 0.84 s | 61 ms | 0.90 s | 0.41 s (48 conns) |
| Johnson Co. buffer(-1) x buffer(-1) | 708,951 | 0.37 s | 64 ms | 0.43 s | 6.6 s (1 conn) |
| LA parcels x soils (area) | 2,743,450 | 3.61 s | 120 ms | 3.73 s | 18.0 s (48 conns) |
| LA parcels x counties (area) | 2,419,854 | 2.85 s | 81 ms | 2.93 s | 17.7 s (48 conns) |
| LA parcels x parcels (area) | 23,232,840 | 2.38 s | 412 ms | 2.79 s | 61.5 s (48 conns) |
| LA addresses x parcels | 14,078,923 | 10.00 s | 110 ms | 10.11 s | 13.3 s (48 conns) |
| LA ST_ClusterIntersecting | 2,418,914 | 2.38 s | 239 ms | 2.62 s | 32.5 s (1 core) |
| LA parcels x parcels, dwithin 5 m | 23,800,370 | 2.38 s | 232 ms | 2.61 s | 79.3 s (48 conns) |
| LA addresses x parcels, dwithin 5 m | 14,992,942 | 10.00 s | 103 ms | 10.10 s | 21.0 s (48 conns) |
| LA buffer(-1) x buffer(-1) | 10,913,792 | 2.38 s | 345 ms | 2.72 s | 14.8 s (48 conns) |
| LA addresses in buffer(parcel, -1) | 13,914,022 | 10.00 s | 87 ms | 10.08 s | 12.1 s (48 conns) |

Load is the cost of the first query only: one LA parcels load (2.4 s) serves all nine LA workloads above (plus
one load each of soils, counties and addresses for the joins that use them), whose queries take 1.7 s together. With the load counted, the GPU is still faster than 48-core PostGIS everywhere except the three
smallest Johnson Co. joins against a separately loaded layer (soils, counties): there the 0.5-1.5 s load is
larger than the whole PostGIS query. Query times through the daemon carry a ~25 ms fixed per-request cost
(a trivial join is 0.5 ms of GPU work and 26 ms end to end: scratch allocation), which is why the Johnson Co.
queries are 50-70 ms against 2-14 ms of GPU time in the tables above. Bbox-query loads (soils, the address
subset) run as a single COPY stream; the address subset spends 7.5 s in a GiST bbox scan of the 161M-row table
(the whole address table loads in 24 s over 16 connections).

Full table (production daemon, three reloads): parcels 5.2-6.3 min (COPY 135-171 s, prep 125-174 s, upload
29-34 s), addresses 24 s, counties 0.5 s.

| Workload | Load | Query | Load + query | PostGIS (extrapolated from LA) |
|---|---|---|---|---|
| parcels x parcels, dwithin 5 m | 5.2-6.3 min | 13.2 s | 5.5-6.5 min | ~70 min (48 conns) |
| parcels x parcels, buffer(-1) x buffer(-1) | 5.2-6.3 min | 46.7 s | 6.0-7.1 min | ~16 min (48 conns) |
| parcels x parcels, intersects | 5.2-6.3 min | 10.5 s | 5.4-6.5 min | |
| addresses x parcels, dwithin 5 m | 5.6-6.7 min | 3.3 s | 5.7-6.7 min | |
| addresses in buffer(parcel, -1) | 5.6-6.7 min | 2.7 s | 5.6-6.7 min | |

For one-off full-table queries the load dominates (the speedup over PostGIS drops to ~2.5x for the buffer
self-join and ~11x for dwithin); it pays off when the resident layers serve repeated queries.

### Precision: FP64 only

All coordinates and all geometric arithmetic are FP64. Lower-precision storage was evaluated and removed:
FP32 missed ~6% of touching parcel neighbours (shared vertices stop being bit-identical) and ~40 boundary
points in LA; FP16/FP8 were both slower and wrong for 0.08% / 8% of pairs. Refinement is latency/compute bound,
so narrower types did not buy speed, and memory is not the constraint (all of `combine.parcel_enriched` is
~64 GB in FP64). The only non-FP64 data are the BVH's float bounding boxes, which are rounded outward and
serve purely as a conservative candidate filter.

## Tested environment

All results above were produced on this configuration (October 2026):

| Component | Version / detail |
|---|---|
| GPU | NVIDIA H200 NVL (GPU index 3), 141 GB HBM3e (143,771 MiB), compute capability 9.0, 400 W limit, max SM clock 1785 MHz, PCIe Gen5 x16, VBIOS 96.00.D9.00.0E |
| NVIDIA driver | 580.178.04 (driver CUDA API 13.0) |
| CUDA toolkit | 12.9 (nvcc V12.9.86, cudart 12.9.79, CCCL/CUB 12.9.27), built with `-arch=sm_90a`, host compiler g++ 13.4.0 |
| Host | 2x Intel Xeon 6747P (192 logical CPUs), 1 TB RAM, Ubuntu 25.10, kernel 6.17.0-8, glibc 2.42 |
| Database | PostgreSQL 18.4 (Debian 18.4-1.pgdg13+1, in Docker), PostGIS 3.6.3, GEOS 3.14.1, PROJ 9.8.1 |
| Extension build | gcc 14.2 inside the `landstats-database` image (PG18 server headers) |
| Daemon client library | libpq 17.10 (host) |

The system nvcc 12.4 and the stock CUDA 12.9 headers do not compile against glibc 2.41+ (`cospi`/`sinpi`/`rsqrt`
exception-specification clash); `scripts/setup_cuda.sh` makes a patched private copy of the 12.9 toolkit.

## Requirements

### Hardware

| | Requirement | Tested |
|---|---|---|
| GPU | NVIDIA, compute capability 8.0+ (warp reductions); built and tuned for Hopper (`sm_90a`). Ampere (`ARCH=sm_80`) compiles but is untested | H200 NVL 141 GB |
| GPU memory | Resident layers: ~16 bytes per vertex + ~100 bytes per polygon (piece) + 32 bytes per point. Join scratch: up to ~30 GB for full-table parcel self-joins with area / buffers, 6-25 GB for intersects / dwithin, ~4 GB for point joins (bounded by `JoinOpts::max_pairs`, 128M candidate pairs per batch) | all 162M parcels = 65 GB, 161M addresses = 4.9 GB, counties 34 MB; LA County parcels 0.6 GB |
| Host RAM | Load staging: ~0.75 KB per parcel row at peak (raw geometry, dedup and subdivision copies) | full parcel load peaked at 119 GB RSS |
| CPU | Any x86_64; host-side prep (dedup, subdivision) is OpenMP-parallel and scales with cores | 2x Xeon 6747P (192 threads): 162M parcels prepared in 2-3 min |

A single county or state fits on any Hopper GPU; the full US parcel table (65 GB resident + up to 30 GB scratch)
needs a 96 GB+ part (H200, GH200) or lower `max_pairs` on an 80 GB H100.

### Software

| Component | Requirement | Tested |
|---|---|---|
| OS | Linux x86_64 | Ubuntu 25.10 (glibc 2.42) |
| NVIDIA driver | R525+ (CUDA 12 minor-version compatibility) | 580.178.04 |
| CUDA toolkit | 12.x with `sm_90a` support (nvcc, CUB/CCCL headers); CUDA <= 12.9 on glibc >= 2.41 needs `scripts/setup_cuda.sh` | 12.9 |
| Host C++ compiler | a GCC the toolkit supports, C++17, OpenMP (`libgomp`) | g++ 13.4 |
| libpq | headers + library for the daemon (`libpq-dev`) | 17.10 |
| PostgreSQL | 15+ server (the extension uses `InitMaterializedSRF`, `MarkGUCPrefixReserved`), with server development headers and PGXS (`postgresql-server-dev-NN`) to build the extension | 18.4 |
| PostGIS | 3.x in the database the layers are loaded from (geometry columns, read as EWKB over binary COPY). The extension itself does not link PostGIS or GEOS | 3.6.3 (GEOS 3.14.1, PROJ 9.8.1) |
| Data | Polygon / MultiPolygon (or collections of them) and Point columns; Z/M are dropped. Distance and buffer functions expect EPSG:4326 lon/lat; intersects / area / relate / cluster work in any single SRID (area in its units) | EPSG:4326 |
| Python (optional) | 3.10+ with `psycopg` 3, `numpy`, `pandas`: only for `scripts/` (export, ground truth, validation, benchmarks) | 3.13 |

## Build

```sh
git clone git@github.com:landstats-us/postgis-cuda.git && cd postgis-cuda

# Only on glibc >= 2.41 with CUDA <= 12.9 (nvcc then fails on cospi/sinpi/rsqrt declarations): make a patched
# private copy of the toolkit in toolchain/cuda; the build scripts pick it up automatically. No root needed.
CUDA_SRC=/usr/local/cuda-12.9 ./scripts/setup_cuda.sh

./scripts/build_daemon.sh    # build/gpujoind  (daemon; links libpq)
./scripts/build_engine.sh    # build/harness   (file-based test / benchmark driver, optional)
cc -O2 -o build/gsp_cat tools/gsp_cat.c   # raw protocol client (optional)
```

The build scripts take `NVCC` (default: `toolchain/cuda/bin/nvcc`, else `nvcc` on `PATH`), `CCBIN` (host
compiler, default `g++-13` when installed), `ARCH` (default `sm_90a`), `PG_CONFIG` (to find libpq) and `OUT`.

The extension is a plain PGXS module (C, no CUDA). Build it against the exact server it will run in:

```sh
make -C ext PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config           # Postgres on this host
# Postgres in Docker: build inside the server's own image so headers and libc match
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD":/src -w /src/ext --entrypoint make <postgres-image>
```

## Install

The daemon (`gpujoind`) owns the GPU and runs on the GPU host; the extension is a thin client inside Postgres
that talks to it over a unix socket or TCP. The daemon reads the layers itself, through its own libpq connection,
so it must point at the same database the extension is installed in.

### 1. Database role and token for the daemon

```sql
create role gpujoind_ro login password '...';      -- or a trust / peer / .pgpass rule
grant pg_read_all_data to gpujoind_ro;              -- loads are read-only (binary COPY, TID range scans)
```

```sh
sudo install -d -m 700 -o gpujoind /etc/gpujoind
head -c 32 /dev/urandom | base64 | sudo tee /etc/gpujoind/token > /dev/null; sudo chmod 600 /etc/gpujoind/token
```

The token is a shared secret: every request must carry it. It is required for `--tcp`, recommended always
(anything that can reach the socket could otherwise ask the daemon to run load queries with its credentials).

### 2. Run the daemon

```sh
CUDA_VISIBLE_DEVICES=0 ./build/gpujoind --conninfo "host=db-host dbname=landstats user=gpujoind_ro" \
    --socket /run/gpu_spatial/gpujoind.sock --token-file /etc/gpujoind/token --threads 16
    # --tcp 172.25.0.1:7433   additionally listen on TCP (e.g. the Docker bridge gateway, for a containerized Postgres)
```

`--threads` is the number of parallel COPY connections per load. Pick the GPU with `CUDA_VISIBLE_DEVICES`; one
daemon serves one GPU and runs one GPU job at a time. As a systemd service:

```ini
# /etc/systemd/system/gpujoind.service
[Unit]
Description=gpujoind (GPU spatial joins for gpu_spatial)
After=network-online.target

[Service]
User=gpujoind
Environment=CUDA_VISIBLE_DEVICES=0
RuntimeDirectory=gpu_spatial
RuntimeDirectoryMode=0755
ExecStart=/opt/postgis-cuda/build/gpujoind --conninfo "host=db-host dbname=landstats user=gpujoind_ro" \
    --socket /run/gpu_spatial/gpujoind.sock --token-file /etc/gpujoind/token --threads 16
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Layers live in GPU memory only: after a daemon restart they must be loaded again (step 5).

### 3. Install the extension into Postgres

Postgres on the same host:

```sh
sudo make -C ext install PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config
```

Postgres in a Docker container (`database` below): copy the build into the server's directories,

```sh
LIB=$(docker exec database pg_config --pkglibdir); SHARE=$(docker exec database pg_config --sharedir)
docker cp ext/gpu_spatial.so database:$LIB/
docker cp ext/gpu_spatial.control database:$SHARE/extension/
docker cp ext/gpu_spatial--0.1.sql database:$SHARE/extension/
```

or, so it survives the container being recreated, bake it into the image:

```dockerfile
COPY ext/gpu_spatial.so /usr/lib/postgresql/18/lib/
COPY ext/gpu_spatial.control ext/gpu_spatial--0.1.sql /usr/share/postgresql/18/extension/
```

No `shared_preload_libraries` entry or server restart is needed; the library loads on first use.

### 4. Connect Postgres to the daemon

* Same host, unix socket: the Postgres server process must be able to open the socket. For a container,
  bind-mount the socket directory (`-v /run/gpu_spatial:/run/gpu_spatial`).
* TCP: start the daemon with `--tcp HOST:PORT --token-file ...` on an address the server can reach (for Docker,
  the bridge gateway, e.g. `172.25.0.1`), and keep the port firewalled from everything else.

Then, as a superuser:

```sql
create extension gpu_spatial;                                     -- superuser-only (superuser = true)
alter system set gpu_spatial.daemon = 'unix:/run/gpu_spatial/gpujoind.sock';   -- or 'tcp:172.25.0.1:7433'
alter system set gpu_spatial.token = '<contents of /etc/gpujoind/token>';
select pg_reload_conf();
select gpu_status();                                              -- device, free memory, loaded layers
```

Both settings are superuser-only (`gpu_spatial.token` is also hidden from `SHOW ALL`), so sessions use them
without being able to change or read them. The default address is `unix:/var/run/gpu_spatial/gpujoind.sock`.
`gpu_load`, `gpu_load_query` and `gpu_unload` are revoked from `PUBLIC`; the query functions are open to all roles.

### 5. Load layers

```sql
select gpu_load('parcels', 'combine.parcel_enriched', id_column => 'landrecid', subdivide => 256);  -- dedup on
select gpu_load('counties', 'tiger_data.county_4326', id_column => 'gid', subdivide => 64);
select gpu_load('addresses', 'combine.overturemaps_address', id_column => 'id');                    -- point layer
select * from gpu_layers();
```

Uninstall with `drop extension gpu_spatial;` and by removing the three files.

## Usage

```sql
-- ST_Intersects + ST_Area(ST_Intersection) (area in SRID units, deg^2 for 4326)
select a_id::uuid as landrecid, b_id::int as gid, area from gpu_join('parcels', 'counties');
select * from gpu_join('parcels', 'parcels', with_area => true, self_pairs => 'distinct');
select * from gpu_intersects('addresses', 'parcels');                 -- point in polygon
select * from gpu_cluster_intersecting('parcels');                    -- (id, cluster_id)
select * from gpu_relate('parcels', 'counties');                      -- (a_id, b_id, DE-9IM)
select * from gpu_relate('parcels', 'parcels', 'T*T***T**', 'distinct');   -- ST_Relate(a, b, pattern): overlaps
select gpu_join_stats('parcels', 'parcels', true, 'distinct');        -- rows + area_sum, no row delivery
select * from gpu_dwithin('parcels', 'parcels', 5, 'distinct');       -- ST_DWithin(geography), meters
select * from gpu_dwithin('addresses', 'parcels', 25);
-- st_intersects(st_buffer(a.geom::geography, -1)::geometry, st_buffer(b.geom::geography, -1)::geometry):
select * from gpu_intersects_buffered('parcels', 'parcels', -1, -1, 'distinct');
select * from gpu_intersects_buffered('parcels', 'addresses', -1, 0); -- points inside parcels eroded by 1 m
select gpu_last_stats();                                              -- timings of the last call
```

`gpu_load` reads partitioned tables in parallel (per leaf partition and block range, TID range scans).
The daemon loads from its own `--conninfo`; it must point at the same database as the extension.
Load/unload functions are superuser-only. The `filter` argument of `gpu_load` is raw SQL.

### SQL API

| Function | Returns |
|---|---|
| `gpu_load(layer, rel, id_column='id', geom_column='geom', filter=NULL, subdivide=0, dedup=true)` | `jsonb` load summary (superuser) |
| `gpu_load_query(layer, query, subdivide=0, dedup=true)` | `jsonb` (query returns `(id, geom)`; superuser) |
| `gpu_unload(layer)` | `bool` |
| `gpu_join(a, b, with_area=true, self_pairs='all')` | `(a_id, b_id, area)`: ST_Intersects pairs, area = ST_Area(ST_Intersection) |
| `gpu_intersects(a, b, self_pairs='all')` / `gpu_intersection_area(...)` | wrappers of `gpu_join` |
| `gpu_join_stats(a, b, with_area=true, self_pairs='all')` | `jsonb` row count + area sum, no rows |
| `gpu_relate(a, b, pattern=NULL, self_pairs='all')` | `(a_id, b_id, relate)`: DE-9IM of every intersecting pair, optionally pattern-filtered |
| `gpu_cluster_intersecting(layer)` | `(id, cluster_id)`, like `ST_ClusterIntersectingWin` |
| `gpu_dwithin(a, b, distance, self_pairs='all')` | `(a_id, b_id)`: `ST_DWithin(a::geography, b::geography, distance)`, meters; point layers on either side |
| `gpu_intersects_buffered(a, b, buffer_a, buffer_b, self_pairs='all')` | `(a_id, b_id)`: `ST_Intersects(ST_Buffer(a::geography, buffer_a), ST_Buffer(b::geography, buffer_b))`, meters; both <= 0 (0 = unbuffered, points must be 0) or both >= 0 (as `gpu_dwithin(a, b, buffer_a + buffer_b)`) |
| `gpu_dwithin_stats(...)`, `gpu_intersects_buffered_stats(...)` | `jsonb` row count and timings, no rows |
| `gpu_layers()`, `gpu_layers_json()`, `gpu_status()`, `gpu_last_stats()` | introspection |

GUCs (superuser): `gpu_spatial.daemon` (`unix:/path` or `tcp:host:port`), `gpu_spatial.token`.
Point layers join with polygon layers in either order (point in polygon; boundary counts).
`gpu_relate` returns intersecting pairs only: patterns satisfied by disjoint pairs (`FF*FF****`) are not
enumerable at this scale.

### Layer freshness and load time

A layer is a snapshot taken by `gpu_load`; it is not synced with later changes to the base table. Re-run
`gpu_load` with the same name to refresh it (the old copy keeps serving queries until the new one is ready).
A full load of `combine.parcel_enriched` (162M rows) takes ~6.3 min: ~171 s COPY over 16 connections (the
row-store heap is read in full: 233 GB since the 2026-10-16 `VACUUM FULL`, down from 1.23 TB when ~45% of row
versions were dead, which took ~270 s), ~171 s host prep (dedup, subdivision) and ~34 s upload + BVH. Prep and
upload are the next things to parallelize.

## Validate

```sh
./scripts/validate.sh            # GPU vs every PostGIS ground-truth CSV in data/
```

Ground truth is produced with `scripts/truth_sql.sh` (N concurrent connections, hash-sharded; `CONN` selects the
server, default the production server as `claude`). The distance and buffer truths use `st_dwithin(a.geom::geography,
b.geom::geography, d)` and, for buffers, a table of `st_buffer(geom::geography, -1)::geometry` with a GiST index
joined by `st_intersects`; the LA ones were run on the dev instance (same PostGIS / GEOS / PROJ build, same data).

## Known limitations

* Ids are returned as text; cast in SQL (`a_id::uuid`).
* `ST_Intersection` geometries are not materialized; the intersection *area* is computed directly.
* Results go through a Postgres set-returning function; for tens of millions of rows the backend
  (tuplestore) is the bottleneck. Aggregate in SQL over the result or keep outputs selective.
* Ring orientation is normalized at load from an accurate signed area (relative to the ring's first vertex,
  compensated). On absolute lon/lat coordinates the products cancel catastrophically: 20 sliver rings in LA
  (down to 4e-14 deg^2) had come out inside-out, the source of an earlier 5 m^2 area error. Invalid geometries
  (self-intersections) follow the even-odd rule, which can differ from GEOS.
* Buffers: GEOS simplifies buffer input first (concavities shallower than r/100 are dropped), which is not
  emulated, and snap-rounds; pairs whose eroded polygons come within ~r/100 of touching can differ from
  PostGIS (none did on the parcel tests). Buffers of opposite sign (one side grown, the other eroded) are not
  supported. Positive buffers are answered as `ST_DWithin`, i.e. with round buffers, where GEOS's chord
  approximation is up to 0.5% smaller at convex corners. Polar areas (|lat| > 70, where `_ST_BestSRID` picks
  Lambert azimuthal) are buffered without the projection's distortion.
* `ST_DWithin` treats edges as straight in lon/lat except near the threshold, where it follows PostGIS's
  great-circle model; geography containment (distance 0) uses the planar polygon, which differs only within
  the edge's sagitta (< 0.1 mm for 100 m edges) of the boundary.
* One GPU job at a time per daemon; layers live until `gpu_unload` or daemon restart, and are not synced with
  the base table (see "Layer freshness").
* `ST_Relate` uses exact arithmetic on the stored coordinates; GEOS nodes segments, rounding intersection points,
  so where a vertex lies within ~1 ulp of a neighbour's edge the two can disagree on the BB dimension (0 vs 1)
  or see a sub-ulp sliver differently. Agreement with PostGIS: parcels x counties and addresses x parcels 100%;
  Johnson Co. parcels x parcels 99.9994% (7 of 1.19M); LA parcels x parcels 99.30% exact matrices and 99.994% at
  the T/F level that ST_Touches/Overlaps/Equals/Contains/Within/Covers test (1,366 of 23.2M differ, mostly
  neighbours that GEOS calls touching and exact arithmetic finds overlapping by a sub-ulp sliver).
  `RefineParams::snap_half_ulp` treats vertices within half an ulp of a line as on it (not GEOS-exact either).
