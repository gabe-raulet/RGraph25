#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

#include "fmt/core.h"
#include "fmt/ranges.h"
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
#include <mpi.h>

#include "utils.h"
#include "ctree.h"
#include "voronoi.h"

Distance distance;

int main(int argc, char *argv[])
{
    int myrank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const char *points_fname;
    Real epsilon;
    Index num_sites;

    Index leaf_size = 50;
    Real covering_factor = 1.8;
    bool random_sites = false;
    const char *graph_fname = NULL;

    PointVector mypoints;
    Index myoffset, totsize;
    double t, maxtime, tottime = 0;

    auto usage = [&] (int err, bool isroot)
    {
        if (isroot)
        {
            fprintf(stderr, "Usage: %s [options] <points> <epsilon> <num_sites>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
            fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
            fprintf(stderr, "         -o FILE   graph output file\n");
            fprintf(stderr, "         -R        choose sites randomly\n");
            fprintf(stderr, "         -h        help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:o:Rh")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') graph_fname = optarg;
        else if (c == 'R') random_sites = true;
        else if (c == 'h') usage(0, !myrank);
    }

    if (argc - optind < 3)
    {
        if (!myrank) fmt::print(stderr, "[err::{}] missing argument(s)\n", __func__);
        usage(1, !myrank);
    }

    points_fname = argv[optind++];
    epsilon = atof(argv[optind++]);
    num_sites = atoi(argv[optind]);

    read_fvecs(mypoints, myoffset, totsize, points_fname, MPI_COMM_WORLD);

    assert((myoffset == myrank*(totsize/nprocs)));

    Index n = mypoints.size();
    Index m = num_sites;

    /*
     * Build Voronoi diagram
     */

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    VoronoiDiagram diagram(mypoints.data(), mypoints.size(), myoffset, MPI_COMM_WORLD);

    if (random_sites) diagram.build_random_diagram(m);
    else diagram.build_greedy_diagram(m);

    t += MPI_Wtime();
    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={}]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest());

    MPI_Finalize();
    return 0;
}
