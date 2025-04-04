import numpy as np
from snnpy import *
import time
import sys
import os

def read_file(fname):
    fsize = os.path.getsize(fname)
    with open(fname, "rb") as f:
        d = int.from_bytes(f.read(4), byteorder="little")
        n = fsize // (4 * (d + 1))
        points = np.zeros((n, d), dtype=np.float32)
        f.seek(0, 0)
        for i in range(n):
            assert d == int.from_bytes(f.read(4), byteorder="little")
            points[i] = np.frombuffer(f.read(4*d), dtype=np.float32, count=d)
    return points

def main(fname, epsilon):

    points = read_file(fname)

    t = -time.perf_counter()
    ds = build_snn_model(points)
    t += time.perf_counter()
    index_time = t

    n_points = len(points)
    n_edges = 0

    t = -time.perf_counter()

    for j in range(n_points):
        ind = ds.query_radius(points[j], epsilon)
        n_edges += len(ind)

    t += time.perf_counter()
    graph_time = t

    print(f"time={index_time + graph_time:.3f}\tindex_time={index_time:.3f}\tgraph_time={graph_time:.3f}\tn_edges={n_edges}\tdensity={n_edges/n_points:.3f}")

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <points> <epsilon>")
        sys.exit(0)
    else:
        main(sys.argv[1], float(sys.argv[2]))

#  points = read_file("p10k.fvecs")

#  t = -time.perf_counter()
#  snn_model = build_snn_model(points)
#  t += time.perf_counter()

#  print(f"[time={t:.3f}] snn index")

#  n_points = len(points)
#  n_edges = 0

#  t = -time.perf_counter()

#  for j in range(n_points):
    #  ind = snn_model.query_radius(points[j], 2.75)
    #  n_edges += len(ind)

#  t += time.perf_counter()

#  print(f"[time={t:.3f}] snn find neighbors [edges={n_edges},density={n_edges/n_points:.3f}]")
