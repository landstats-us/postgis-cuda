/* gpujoind wire protocol (shared by the daemon and the Postgres extension). Little endian.
 *
 * request : u32 magic GSP_MAGIC, u32 op, u32 nargs, nargs x { u32 len, bytes }
 *           args[0] is always the auth token (empty if the daemon runs without --token); the op args follow
 * response: u32 status (GSP_OK | GSP_ERR); on GSP_ERR: u32 len, message
 *   GSP_OP_LOAD / LOAD_QUERY / DROP / STATUS / LIST : u32 len, JSON text
 *   GSP_OP_JOIN / CLUSTER / RELATE / DWITHIN : stream of frames { u32 nrows, rows... } ended by nrows == 0, then u32 len, JSON stats;
 *                           nrows == GSP_FRAME_ERR is followed by u32 len, message (error mid-stream)
 *   join row    : u16 la, a_id, u16 lb, b_id, f64 area (NaN when area was not requested; always NaN for DWITHIN)
 *   cluster row : u16 la, id, u32 cluster
 *   relate row  : u16 la, a_id, u16 lb, b_id, char[9] DE-9IM matrix
 *
 * ops and args (all args are UTF-8 strings):
 *   LOAD       layer, table, id_col, geom_col, filter, subdivide, dedup      (all geometry is FP64)
 *   LOAD_QUERY layer, sql, subdivide, dedup
 *   DROP       layer
 *   LIST
 *   STATUS
 *   JOIN       layer_a, layer_b, want_area ("1"/"0"), self_pairs ("all" | "distinct")
 *   CLUSTER    layer
 *   RELATE     layer_a, layer_b, pattern ("" = all intersecting pairs), self_pairs ("all" | "distinct")
 *   DWITHIN    layer_a, layer_b, mode, dist_a, dist_b, self_pairs[, "stats"]   (meters, geography semantics)
 *              mode "dwithin": ST_DWithin(a, b, dist_a);
 *              mode "buffer":  ST_Intersects(ST_Buffer(a, dist_a), ST_Buffer(b, dist_b)), both <= 0 or both >= 0
 */
#ifndef GSP_PROTOCOL_H
#define GSP_PROTOCOL_H

#define GSP_MAGIC 0x52505347u /* "GSPR" */
#define GSP_OK 0u
#define GSP_ERR 1u
#define GSP_FRAME_ERR 0xFFFFFFFFu

enum {
  GSP_OP_LOAD = 1,
  GSP_OP_LOAD_QUERY = 2,
  GSP_OP_DROP = 3,
  GSP_OP_LIST = 4,
  GSP_OP_STATUS = 5,
  GSP_OP_JOIN = 6,
  GSP_OP_CLUSTER = 7,
  GSP_OP_RELATE = 8,
  GSP_OP_DWITHIN = 9
};

#endif
