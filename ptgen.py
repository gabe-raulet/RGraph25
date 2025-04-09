import os
import sys
import numpy as np
import math
import time

def write_file(points, fname):
    assert points.dtype == np.float32
    n, d = points.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(d.to_bytes(4, byteorder="little"))
            f.write(points[i].tobytes())

def random_rotation_matrix(d):
    A = np.random.normal(0, 1, (d, d))
    Q, R = np.linalg.qr(A)

    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1

    return Q

def main(n, dim, ambient, fname):
    p1 = np.random.normal(0, 1, (n, dim))
    p2 = np.zeros((n, ambient-dim))
    A = np.hstack([p1,p2])
    R = random_rotation_matrix(ambient)
    points = (A@R).astype(np.float32)
    write_file(points, fname)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <n> <dim> <ambient> <fname>")
        sys.exit(1)
    else:
        sys.exit(main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
