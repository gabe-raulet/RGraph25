#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "fmt/core.h"
#include <array>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <omp.h>
#include "utils.h"

Index read_integer(char *str);

int main(int argc, char *argv[])
{
    Index size;
    PointVector points;
    const char *fname = NULL;

    int seed = -1;

    auto usage = [&] (int err)
    {
        fprintf(stderr, "Usage: %s [options] <size> <filename>\n", argv[0]);
        fprintf(stderr, "Options: -s INT   ptgen rng seed [%d]\n", seed);
        fprintf(stderr, "         -h       help message\n");
        exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "s:h")) >= 0)
    {
        if      (c == 's') seed = atoi(optarg);
        else if (c == 'h') usage(0);
    }

    if (argc - optind < 2)
    {
        fmt::print(stderr, "[err::{}] missing argument(s)\n", __func__);
        usage(1);
    }

    size = read_integer(argv[optind++]);
    fname = argv[optind];

    if (seed < 0)
    {
        std::random_device rd;
        seed = rd();
        seed = seed < 0? -seed : seed;
    }

    double t;

    t = -omp_get_wtime();

    points.resize(size);
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<Real> dist{0., 1.};

    for (Point& p : points)
    {
        std::generate(p.begin(), p.end(), [&] () { return dist(gen); });
    }

    t += omp_get_wtime();

    fmt::print("[time={:.3f}] generated {} points [dimension={},seed={}]\n", t, size, DIM_SIZE, seed);

    t = -omp_get_wtime();
    write_fvecs(points, fname);
    t += omp_get_wtime();

    fmt::print("[time={:.3f}] wrote points to file '{}'\n", t, fname);

    return 0;
}

Index read_integer(char *str)
{
    double x;
    char *p;

    x = strtod(str, &p);

    if      (toupper(*p) == 'K') x *= (1LL << 10);
    else if (toupper(*p) == 'M') x *= (1LL << 20);
    else if (toupper(*p) == 'G') x *= (1LL << 30);

    return static_cast<Index>(x + 0.499);
}

