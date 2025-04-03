import numpy as np
from snnpy import *
import time
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

points = read_file("p10k.fvecs")
#  points = read_file("siftsmall/siftsmall_base.fvecs")

t = -time.perf_counter()
snn_model = build_snn_model(points)
t += time.perf_counter()

print(f"[time={t:.3f}] snn index")

n_points = len(points)
n_edges = 0

t = -time.perf_counter()

for j in range(n_points):
    ind = snn_model.query_radius(points[j], 2.75)
    n_edges += len(ind)

t += time.perf_counter()

print(f"[time={t:.3f}] snn find neighbors [edges={n_edges},density={n_edges/n_points:.3f}]")
