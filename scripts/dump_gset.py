#!/usr/bin/env python3
"""Dump a polygon query to the flat .gset format used by the GPU engine test harness.

  dump_gset.py OUT.gset "select <id>, geom from ..."   (first column = id text, second = geometry)

Layout (little endian):
  u32 magic 'GSET', u32 version(1), u64 n, u64 nrings, u64 nverts
  u64 geom_ring_off[n+1]     index into rings
  u64 ring_vert_off[nrings+1] index into vertices (rings are closed: first==last)
  f64 xy[nverts*2]
Ring orientation is normalized: shells CCW, holes CW (the area kernels rely on it).
Also writes OUT.gset.ids (one id per line).
"""
import struct, sys
import numpy as np
import psycopg

def parse_wkb(b):
    """Return list of polygons; polygon = list of rings (np.ndarray Nx2). 2D only (Z/M dropped)."""
    mv = memoryview(b)
    pos = 0
    polys = []

    def rd(fmt, bo):
        nonlocal pos
        v = struct.unpack_from(bo + fmt, mv, pos)
        pos += struct.calcsize(bo + fmt)
        return v

    def geom():
        nonlocal pos
        bo = '<' if mv[pos] == 1 else '>'
        pos += 1
        t, = rd('I', bo)
        flags = t & 0xE0000000
        base = t & 0xFFFF
        hasz = bool(flags & 0x80000000) or (1000 <= base < 2000) or (3000 <= base < 4000)
        hasm = bool(flags & 0x40000000) or (2000 <= base < 3000) or (3000 <= base < 4000)
        if flags & 0x20000000:
            pos += 4  # srid
        base %= 1000
        nd = 2 + hasz + hasm
        if base == 3:
            nr, = rd('I', bo)
            rings = []
            for _ in range(nr):
                np_, = rd('I', bo)
                a = np.frombuffer(mv, dtype=bo + 'f8', count=np_ * nd, offset=pos).reshape(np_, nd)[:, :2]
                pos += np_ * nd * 8
                rings.append(np.array(a, dtype='<f8'))
            polys.append(rings)
        elif base == 6:
            ng, = rd('I', bo)
            for _ in range(ng):
                geom()
        else:
            raise ValueError(f"unsupported wkb type {base}")

    geom()
    return polys

def signed_area(r):
    # relative to the first vertex (absolute lon/lat products cancel catastrophically for sliver rings); exact
    # rational arithmetic when the result is too small to trust the sign
    d = r - r[0]
    x, y = d[:, 0], d[:, 1]
    a = 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    ext = float(np.ptp(d[:, 0]) + np.ptp(d[:, 1]))
    if abs(a) <= 1e-9 * ext * ext:
        from fractions import Fraction
        F = [(Fraction(float(p[0])), Fraction(float(p[1]))) for p in d]
        a = float(sum(F[i][0] * F[i + 1][1] - F[i + 1][0] * F[i][1] for i in range(len(F) - 1)) / 2)
    return a

def main():
    out, sql = sys.argv[1], sys.argv[2]
    dsn = __import__("os").environ.get("PG_DSN", "host=graybase dbname=landstats user=claude")
    ids, geom_ring_off, ring_vert_off, chunks = [], [0], [0], []
    nrings = nverts = 0
    skipped = 0
    with psycopg.connect(dsn) as conn, conn.cursor(name="c") as cur:
        cur.itersize = 20000
        cur.execute("select q.id::text, st_asbinary(q.geom) from (" + sql + ") q(id, geom)")
        for gid, wkb in cur:
            try:
                polys = parse_wkb(bytes(wkb))
            except Exception as e:
                if skipped == 0:
                    print("first parse error:", repr(e), file=sys.stderr)
                skipped += 1
                continue
            rings_out = []
            for rings in polys:
                for k, r in enumerate(rings):
                    if len(r) < 4:
                        continue
                    a = signed_area(r)
                    want_ccw = (k == 0)
                    if (a > 0) != want_ccw:
                        r = r[::-1]
                    rings_out.append(r)
            if not rings_out:
                skipped += 1
                continue
            for r in rings_out:
                nverts += len(r)
                ring_vert_off.append(nverts)
                chunks.append(np.ascontiguousarray(r))
            nrings += len(rings_out)
            geom_ring_off.append(nrings)
            ids.append(str(gid))
    n = len(ids)
    with open(out, "wb") as f:
        f.write(struct.pack("<4sIQQQ", b"GSET", 1, n, nrings, nverts))
        f.write(np.asarray(geom_ring_off, dtype="<u8").tobytes())
        f.write(np.asarray(ring_vert_off, dtype="<u8").tobytes())
        for c in chunks:
            f.write(c.tobytes())
    with open(out + ".ids", "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"{out}: n={n} rings={nrings} verts={nverts} avg_verts={nverts/max(n,1):.1f} skipped={skipped}")

if __name__ == "__main__":
    main()
