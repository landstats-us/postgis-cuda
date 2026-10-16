#!/usr/bin/env python3
"""Dump a point query to the flat .pset format:  dump_pset.py OUT.pset "select <id>, geom from ..."
Layout: u32 magic 'PSET', u32 version(1), u64 n, f64 xy[n*2]; ids in OUT.pset.ids"""
import struct, sys
import numpy as np
import psycopg

def main():
    out, sql = sys.argv[1], sys.argv[2]
    ids, xs, ys = [], [], []
    with psycopg.connect(__import__("os").environ.get("PG_DSN", "host=graybase dbname=landstats user=claude")) as conn, conn.cursor(name="c") as cur:
        cur.itersize = 100000
        cur.execute("select q.id::text, st_x(q.geom), st_y(q.geom) from (" + sql + ") q(id, geom) where q.geom is not null")
        for gid, x, y in cur:
            ids.append(gid); xs.append(x); ys.append(y)
    xy = np.empty((len(ids), 2), dtype="<f8"); xy[:, 0] = xs; xy[:, 1] = ys
    with open(out, "wb") as f:
        f.write(struct.pack("<4sIQ", b"PSET", 1, len(ids))); f.write(xy.tobytes())
    with open(out + ".ids", "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{out}: n={len(ids)}")

if __name__ == "__main__":
    main()
