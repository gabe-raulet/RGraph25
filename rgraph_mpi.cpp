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
    Real cover = 1.8;
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
            fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", cover);
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
        if      (c == 'c') cover = atof(optarg);
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

    diagram.build_replication_tree(cover, leaf_size);

    t += MPI_Wtime();
    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={}]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest());

    /*
     * Compute tree points
     */

    Index num_ghost_points;
    IndexVector sendtreeids, sendtreeptrs, sendghostids, sendghostptrs;

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    diagram.compute_my_tree_points(sendtreeids, sendtreeptrs);
    num_ghost_points = diagram.compute_my_ghost_points(epsilon, sendghostids, sendghostptrs);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] computed ghost points [treepts={},ghostpts={},pts_per_tree={:.1f},ghosts_per_tree={:.1f}]\n", maxtime, totsize, num_ghost_points, totsize/(num_sites+0.0), num_ghost_points/(num_sites+0.0));

    /*
     * Exchange points
     */

    IndexVector assignments(m), mysites, mytreeids, mytreeptrs, myghostids, myghostptrs;
    PointVector mytreepts, myghostpts;

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    for (Index i = 0; i < m; ++i)
    {
        assignments[i] = i % nprocs;
    }

    diagram.exchange_points(sendtreeids, sendtreeptrs, sendghostids, sendghostptrs, assignments, mysites, mytreeids, mytreeptrs, mytreepts);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] exchanged and repacked points alltoall\n", maxtime);

    /*
     * Build ghost trees
     */
    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    Index s = mysites.size();
    std::vector<CoverTree> ghost_trees(s);

    for (Index i = 0; i < s; ++i)
    {
        auto pfirst = mytreepts.begin() + mytreeptrs[i];
        auto plast = mytreepts.begin() + mytreeptrs[i+1];

        auto ifirst = mytreeids.begin() + mytreeptrs[i];
        auto ilast = mytreeids.begin() + mytreeptrs[i+1];

        ghost_trees[i].build(pfirst, plast, ifirst, ilast, cover, leaf_size);
        ghost_trees[i].set_site(mysites[i]);
    }

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] computed ghost trees\n", maxtime);

    MPI_Finalize();
    return 0;
}
