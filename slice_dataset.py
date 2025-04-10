import os
import sys
import numpy as np
import math
import time

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

def write_file(points, fname):
    assert points.dtype == np.float32
    n, d = points.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(d.to_bytes(4, byteorder="little"))
            f.write(points[i].tobytes())


def main(ifname, ofname, start, size):
    points = read_file(ifname)
    np.random.shuffle(points)
    points = points[start:start+size]
    write_file(points, ofname)
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <input> <output> <start> <size>")
        sys.exit(1)
    else:
        sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
