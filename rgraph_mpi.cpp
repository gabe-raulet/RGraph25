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

void rebalance_trees(const CoverTree *send_trees, int sendcount, std::vector<CoverTree>& recv_trees, MPI_Comm comm);

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
    Index rebalance_rate = 10;
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
            fprintf(stderr, "         -n INT    rebalance rate [%d]\n", (int)rebalance_rate);
            fprintf(stderr, "         -o FILE   graph output file\n");
            fprintf(stderr, "         -R        choose sites randomly\n");
            fprintf(stderr, "         -h        help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:o:n:Rh")) >= 0)
    {
        if      (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') graph_fname = optarg;
        else if (c == 'R') random_sites = true;
        else if (c == 'n') rebalance_rate = atoi(optarg);
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

    /*
     * Build epsilon graph
     */

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    IndexVectorVector mygraph;
    IndexVector myids;

    mygraph.reserve(totsize/nprocs);
    myids.reserve(totsize/nprocs);

    Index my_n_edges = 0, n_edges;
    int done = 0;

    do
    {
        const CoverTree *tree = ghost_trees.data();
        Index num_left = ghost_trees.size();

        for (Index i = 0; i < rebalance_rate && num_left > 0; ++i, ++tree, --num_left)
        {
            Index cellsize = diagram.get_cell_size(tree->get_site());
            my_n_edges += tree->graph_query(mygraph, myids, cellsize, epsilon);
        }

        done = !!(num_left == 0);

        MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (!done)
        {
            std::vector<CoverTree> recv_trees;
            rebalance_trees(tree, num_left, recv_trees, MPI_COMM_WORLD);
            std::swap(ghost_trees, recv_trees);
        }

    } while (!done);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    MPI_Reduce(&my_n_edges, &n_edges, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    Real density, sparsity;
    density = (n_edges+0.0)/totsize;
    sparsity = density / totsize;

    if (!myrank) fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},sparsity={:.3f},edges={}]\n", maxtime, density, sparsity, n_edges);

    MPI_Finalize();
    return 0;
}

void rebalance_trees(const CoverTree *send_trees, int sendcount, std::vector<CoverTree>& recv_trees, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    static std::random_device rd;
    static std::default_random_engine gen(rd());

    std::uniform_int_distribution<int> dist{0, nprocs-1};

    std::vector<int> assignments(sendcount);
    std::generate(assignments.begin(), assignments.end(), [&]() { return dist(gen); });

    std::vector<int> sendcounts(nprocs, 0), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);

    for (int i = 0; i < sendcount; ++i)
    {
        int dest = assignments[i];
        sendcounts[dest]++;
    }

    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), static_cast<int>(0));

    IndexVector send_sites(sendcounts.back() + sdispls.back());
    std::vector<int> send_sizes(sendcounts.back() + sdispls.back());
    int tot_send_size = 0;

    std::vector<int> ptrs = sdispls;

    for (int i = 0; i < sendcount; ++i)
    {
        int dest = assignments[i];
        int loc = ptrs[dest]++;

        send_sites[loc] = send_trees[i].get_site();
        send_sizes[loc] = send_trees[i].get_packed_bufsize();
        tot_send_size += send_sizes[loc];
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));

    IndexVector recv_sites(recvcounts.back() + rdispls.back());
    std::vector<int> recv_sizes(recvcounts.back() + rdispls.back());

    MPI_Alltoallv(send_sites.data(), sendcounts.data(), sdispls.data(), MPI_INT64_T,
                  recv_sites.data(), recvcounts.data(), rdispls.data(), MPI_INT64_T, MPI_COMM_WORLD);

    MPI_Alltoallv(send_sizes.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                  recv_sizes.data(), recvcounts.data(), rdispls.data(), MPI_INT, MPI_COMM_WORLD);

    int recvcount = recv_sites.size();
    int tot_recv_size = std::accumulate(recv_sizes.begin(), recv_sizes.end(), static_cast<int>(0));

    std::vector<char> sendbuf(tot_send_size);
    std::vector<MPI_Request> send_reqs(sendcount), recv_reqs(recvcount);

    char *sendptr = sendbuf.data();

    for (int i = 0; i < sendcount; ++i)
    {
        int dest = assignments[i];
        int bufsize = send_trees[i].pack_tree(sendptr, comm);
        MPI_Isend(sendptr, bufsize, MPI_PACKED, dest, static_cast<int>(send_trees[i].get_site()), comm, &send_reqs[i]);
        sendptr += bufsize;
    }

    recv_trees.clear();

    std::vector<char> recvbuf;

    recv_trees.resize(recvcount);
    recvbuf.resize(tot_recv_size);

    char *recvptr = recvbuf.data();

    for (int i = 0; i < recvcount; ++i)
    {
        MPI_Irecv(recvptr, recv_sizes[i], MPI_PACKED, MPI_ANY_SOURCE, static_cast<int>(recv_sites[i]), comm, &recv_reqs[i]);
        recvptr += recv_sizes[i];
    }

    recvptr = recvbuf.data();

    for (int i = 0; i < recvcount; ++i)
    {
        MPI_Wait(&recv_reqs[i], MPI_STATUS_IGNORE);
        recv_trees[i].unpack_tree(recvptr, recv_sizes[i], comm);
        recvptr += recv_sizes[i];
    }

    MPI_Waitall(sendcount, send_reqs.data(), MPI_STATUSES_IGNORE);
}
