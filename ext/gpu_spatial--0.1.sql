\echo Use "CREATE EXTENSION gpu_spatial" to load this file. \quit

-- Load (or replace) a resident GPU layer from a table (FP64 coordinates). Partitioned tables are read in parallel
-- per partition and block range. id_column is returned as text by the join functions. subdivide > 0 splits polygons
-- with more vertices into pieces (256 recommended for parcels); dedup stores identical geometries once.
CREATE FUNCTION gpu_load(layer text, rel regclass, id_column text DEFAULT 'id', geom_column text DEFAULT 'geom',
                         filter text DEFAULT NULL, subdivide int DEFAULT 0, dedup bool DEFAULT true)
RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_load' LANGUAGE C VOLATILE;

-- Load a layer from an arbitrary query returning (id, geom).
CREATE FUNCTION gpu_load_query(layer text, query text, subdivide int DEFAULT 0, dedup bool DEFAULT true)
RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_load_query' LANGUAGE C VOLATILE;

CREATE FUNCTION gpu_unload(layer text) RETURNS bool AS 'MODULE_PATHNAME', 'gpu_unload' LANGUAGE C VOLATILE STRICT;
CREATE FUNCTION gpu_status() RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_status' LANGUAGE C VOLATILE;
CREATE FUNCTION gpu_layers_json() RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_layers_json' LANGUAGE C VOLATILE;
CREATE FUNCTION gpu_last_stats() RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_last_stats' LANGUAGE C VOLATILE;

CREATE FUNCTION gpu_layers(OUT layer text, OUT kind text, OUT rows bigint, OUT "distinct" bigint,
                           OUT gpu_geoms bigint, OUT vertices bigint, OUT subdivide int, OUT gpu_mb float8,
                           OUT copy_s float8, OUT prepare_s float8, OUT upload_s float8, OUT source text)
RETURNS SETOF record LANGUAGE sql AS $$
  SELECT layer, kind, rows, "distinct", gpu_geoms, vertices, subdivide, gpu_mb, copy_s, prepare_s, upload_s, source
  FROM jsonb_to_recordset(gpu_layers_json()) AS x(layer text, kind text, rows bigint, "distinct" bigint,
       gpu_geoms bigint, vertices bigint, subdivide int, gpu_mb float8, copy_s float8, prepare_s float8, upload_s float8, source text)
$$;

-- Spatial join of two resident layers: every (a, b) with ST_Intersects(a.geom, b.geom), and
-- area = ST_Area(ST_Intersection(a.geom, b.geom)) when with_area. Point layers give point-in-polygon joins.
-- self_pairs = 'distinct' (only when a = b) returns each unordered pair once (a_id ordered before b_id by load order)
-- and drops a row joined with itself.
CREATE FUNCTION gpu_join(a text, b text, with_area bool DEFAULT true, self_pairs text DEFAULT 'all',
                         OUT a_id text, OUT b_id text, OUT area float8)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'gpu_join' LANGUAGE C VOLATILE STRICT ROWS 1000000;

-- Run a join on the GPU without returning rows: {"rows": ..., "area_sum": ...} plus timings.
CREATE FUNCTION gpu_join_stats(a text, b text, with_area bool DEFAULT true, self_pairs text DEFAULT 'all')
RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_join_stats' LANGUAGE C VOLATILE STRICT;

CREATE FUNCTION gpu_intersects(a text, b text, self_pairs text DEFAULT 'all', OUT a_id text, OUT b_id text)
RETURNS SETOF record LANGUAGE sql AS $$ SELECT a_id, b_id FROM gpu_join(a, b, false, self_pairs) $$ ROWS 1000000;

CREATE FUNCTION gpu_intersection_area(a text, b text, self_pairs text DEFAULT 'all', OUT a_id text, OUT b_id text, OUT area float8)
RETURNS SETOF record LANGUAGE sql AS $$ SELECT a_id, b_id, area FROM gpu_join(a, b, true, self_pairs) $$ ROWS 1000000;

-- ST_Relate for every intersecting pair of two resident layers: the DE-9IM matrix of relate(a.geom, b.geom).
-- With a pattern (T F * 0 1 2), only pairs matching it are returned, like ST_Relate(a, b, pattern). Disjoint pairs
-- (matrix FF*FF****) are never returned.
CREATE FUNCTION gpu_relate(a text, b text, pattern text DEFAULT NULL, self_pairs text DEFAULT 'all',
                           OUT a_id text, OUT b_id text, OUT relate text)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'gpu_relate' LANGUAGE C VOLATILE ROWS 1000000;

-- Pairs within distance meters of each other: ST_DWithin(a.geom::geography, b.geom::geography, distance), from
-- the boundaries (WGS84 ellipsoid; edges straight in lon/lat). Point layers work on either side.
CREATE FUNCTION gpu_dwithin(a text, b text, distance float8, self_pairs text DEFAULT 'all', OUT a_id text, OUT b_id text)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'gpu_dwithin' LANGUAGE C VOLATILE STRICT ROWS 1000000;
CREATE FUNCTION gpu_dwithin_stats(a text, b text, distance float8, self_pairs text DEFAULT 'all')
RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_dwithin_stats' LANGUAGE C VOLATILE STRICT;

-- Pairs whose buffers intersect, meters:
--   ST_Intersects(ST_Buffer(a.geom::geography, buffer_a)::geometry, ST_Buffer(b.geom::geography, buffer_b)::geometry)
-- Negative buffers erode (e.g. -1, -1: pairs that overlap by more than a sliver, not just touch); 0 leaves a side as is
-- (points must be 0). No buffer polygons are built. Both buffers must be <= 0, or both >= 0 (then computed as
-- gpu_dwithin(a, b, buffer_a + buffer_b), which uses round buffers where GEOS uses 8 segments per quarter circle).
CREATE FUNCTION gpu_intersects_buffered(a text, b text, buffer_a float8, buffer_b float8, self_pairs text DEFAULT 'all',
                                        OUT a_id text, OUT b_id text)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'gpu_intersects_buffered' LANGUAGE C VOLATILE STRICT ROWS 1000000;
CREATE FUNCTION gpu_intersects_buffered_stats(a text, b text, buffer_a float8, buffer_b float8, self_pairs text DEFAULT 'all')
RETURNS jsonb AS 'MODULE_PATHNAME', 'gpu_intersects_buffered_stats' LANGUAGE C VOLATILE STRICT;

-- Window-function style ST_ClusterIntersecting: a cluster number per row (connected components of ST_Intersects).
CREATE FUNCTION gpu_cluster_intersecting(layer text, OUT id text, OUT cluster_id int)
RETURNS SETOF record AS 'MODULE_PATHNAME', 'gpu_cluster_intersecting' LANGUAGE C VOLATILE STRICT ROWS 1000000;

REVOKE ALL ON FUNCTION gpu_load(text, regclass, text, text, text, int, bool) FROM PUBLIC;
REVOKE ALL ON FUNCTION gpu_load_query(text, text, int, bool) FROM PUBLIC;
REVOKE ALL ON FUNCTION gpu_unload(text) FROM PUBLIC;
