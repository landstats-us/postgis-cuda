# postgis-cuda: GPU spatial joins for PostgreSQL/PostGIS (CUDA, Hopper / sm_90)

## What it does

postgis-cuda runs spatial joins on an NVIDIA GPU instead of the CPU. Geometry layers are loaded once
from Postgres into GPU memory, and every query against them — intersects pairs, intersection areas,
DE-9IM relations, clustering, distance and buffer joins — runs on the GPU in milliseconds.

Supported operations, each a drop-in equivalent of a PostGIS function:

| postgis-cuda | PostGIS equivalent |
|---|---|
| `gpu_join` / `gpu_intersects` | `ST_Intersects`, `ST_Area(ST_Intersection(a, b))` |
| `gpu_relate` | `ST_Relate` (full DE-9IM, with pattern matching) |
| `gpu_intersects('points', 'polygons')` | point in polygon |
| `gpu_cluster_intersecting` | `ST_ClusterIntersecting` |
| `gpu_dwithin` | `ST_DWithin(a::geography, b::geography, d)` — meters |
| `gpu_intersects_buffered` | `ST_Intersects(ST_Buffer(a::geography, -ra), ST_Buffer(b::geography, -rb))` — without building buffers |

It is two pieces:

* **`gpujoind`** — a CUDA daemon that owns one GPU, loads layers from the database over libpq, and
  keeps them resident in GPU memory.
* **`gpu_spatial`** — a thin Postgres extension (plain C: no CUDA, no PostGIS linkage) that turns SQL
  function calls into requests to the daemon over a unix socket or TCP and streams the results back.

```
 Postgres backend (gpu_spatial.so)  --unix socket / tcp-->  gpujoind (CUDA, owns one GPU)
   gpu_load(...)            ---------------------------->  parallel COPY (binary) from Postgres
   gpu_join(a, b)           <---- streamed row frames ----  layers resident in GPU memory
   gpu_cluster_intersecting                                 LBVH + exact refinement kernels
```

Results are validated pair-for-pair against PostGIS 3.6 on parcel data, in FP64 — the GPU returns the
same rows PostGIS does, only far faster.

## Benefits

* **50x–2,000x faster joins.** LA parcels x counties (2.4M rows): 17.7 s with 48 PostGIS connections
  vs 7.7 ms on the GPU (~2,300x). LA parcels x parcels (23M rows): 61.5 s vs 297 ms (~207x).
  Johnson Co. parcels within 5 m of each other (1.2M rows): 76.2 s vs 1.9 ms.
* **Exact, not approximate.** All geometry is FP64 and every benchmarked result matches PostGIS
  row-for-row (0 missing, 0 extra pairs).
* **Geography semantics in SQL, at GPU speed.** Meter-based `ST_DWithin` and negative-buffer
  intersects on lon/lat data, matching PostGIS's geography behavior — no buffer materialization.
* **Scales to national data.** All 162M US parcels + 161M addresses resident on one H200 (70 GB):
  the full parcel self-`ST_DWithin` (1.3 billion pairs) in 13 s, where PostGIS needs on the order of
  an hour and a half.
* **Load once, query many.** A layer load (e.g. 2.4 s for LA's 2.4M parcels) is a one-time cost; every
  later join reuses the resident copy. Even counting the load, the GPU beats 48-core PostGIS on nearly
  every tested workload.
* **Safe in the database.** The extension is a client only; the daemon reads with a read-only role, and
  a shared token gates every request.

## System requirements

### Hardware

| | Requirement |
|---|---|
| GPU | NVIDIA, compute capability 8.0+; built and tuned for Hopper (`sm_90a`, tested on H200 NVL 141 GB). Ampere compiles (`ARCH=sm_80`) but is untested |
| GPU memory | ~16 bytes per vertex + ~100 bytes per polygon + 32 bytes per point, plus join scratch (6–30 GB for full-table parcel self-joins). A county or state fits on any Hopper GPU; the full US parcel table (65 GB + scratch) needs a 96 GB+ part |
| Host RAM | ~0.75 KB per parcel row at peak during a load (119 GB for the full parcel table) |
| CPU | Any x86_64; load preparation is OpenMP-parallel |

### Software

| | Requirement |
|---|---|
| OS | Linux x86_64 (tested: Ubuntu 25.10) |
| GPU stack | NVIDIA driver R525+, CUDA 12.x toolkit, GCC with C++17 + OpenMP |
| PostgreSQL | 15+ with server development headers (PGXS); tested on 18.4 |
| PostGIS | 3.x in the source database (the extension itself does not link PostGIS) |
| Data | Polygon/MultiPolygon and Point columns, EWKB-compatible; distance and buffer functions expect EPSG:4326 |
| Python | Optional, for the benchmark/validation scripts only (3.10+, psycopg 3) |

One daemon per GPU, running one job at a time; Postgres and the daemon can be on the same host (unix
socket) or different hosts (TCP).

## Install

Full walkthrough — systemd unit, Docker deployment, security notes — is in
[docs/technical-details.md](docs/technical-details.md#install). Summary:

**1. Build** (see [docs/technical-details.md](docs/technical-details.md#build) for options):

```sh
git clone git@github.com:landstats-us/postgis-cuda.git && cd postgis-cuda
./scripts/build_daemon.sh                                   # build/gpujoind
make -C ext PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config  # the extension, against your server
```

**2. Give the daemon a read-only database role and a shared secret:**

```sql
create role gpujoind_ro login password '...';
grant pg_read_all_data to gpujoind_ro;
```

```sh
head -c 32 /dev/urandom | base64 | sudo tee /etc/gpujoind/token   # chmod 600
```

**3. Run the daemon** on the GPU host (as a systemd service in production):

```sh
CUDA_VISIBLE_DEVICES=0 ./build/gpujoind --conninfo "host=db-host dbname=landstats user=gpujoind_ro" \
    --socket /run/gpu_spatial/gpujoind.sock --token-file /etc/gpujoind/token
```

**4. Install the extension and point Postgres at the daemon:**

```sh
sudo make -C ext install PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config
```

```sql
create extension gpu_spatial;
alter system set gpu_spatial.daemon = 'unix:/run/gpu_spatial/gpujoind.sock';  -- or 'tcp:host:port'
alter system set gpu_spatial.token = '<contents of /etc/gpujoind/token>';
select pg_reload_conf();
select gpu_status();     -- device, free memory, loaded layers
```

**5. Load a layer and join:**

```sql
select gpu_load('parcels', 'combine.parcel_enriched', id_column => 'landrecid', subdivide => 256);
select gpu_load('counties', 'tiger_data.county_4326', id_column => 'gid', subdivide => 64);

select a_id, b_id, area from gpu_join('parcels', 'counties');            -- intersects + area
select * from gpu_dwithin('parcels', 'parcels', 5, 'distinct');          -- within 5 m
```

Layers are snapshots: reload with `gpu_load` to refresh. After a daemon restart, layers must be loaded
again.

## Documentation

* [Technical details](docs/technical-details.md) — algorithms (LBVH, exact FP64 refinement, buffer
  erosion), full benchmark results including load times, tested environment, complete build/install
  guides, full SQL API reference, validation methodology, and known limitations.
* Repository layout: `engine/` (CUDA kernels), `daemon/` (`gpujoind`), `ext/` (the Postgres
  extension), `scripts/` (build, benchmarks, validation), `tools/` (raw protocol client).

## License

PostgreSQL License — see [LICENSE](LICENSE).
