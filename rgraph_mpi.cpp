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

void rebalance_trees(const GhostTree *send_trees, int sendcount, std::vector<GhostTree>& recv_trees, MPI_Comm comm);

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
    Index myoffset, totsize, mydistcomps = 0, tot_distcomps = 0, distcomps;
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

    if (random_sites) diagram.build_random_diagram(m, mydistcomps);
    else diagram.build_greedy_diagram(m, mydistcomps);

    diagram.build_replication_tree(cover, leaf_size, mydistcomps);

    #ifdef STATS
    t += MPI_Wtime();
    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    tot_distcomps += distcomps;

    double voronoi_time = maxtime;
    //if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={},distcomps={:.1f}M]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest(), distcomps/1000000.);
    #endif

    /*
     * Compute tree points
     */

    Index num_ghost_points;
    IndexVector sendtreeids, sendtreeptrs, sendghostids, sendghostptrs;
    mydistcomps = 0;

    #ifdef STATS
    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();
    #endif

    diagram.compute_my_tree_points(sendtreeids, sendtreeptrs);
    num_ghost_points = diagram.compute_my_ghost_points(epsilon, sendghostids, sendghostptrs, mydistcomps);

    #ifdef STATS
    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    tot_distcomps += distcomps;

    double tree_points_time = maxtime;
    Index treepts_o = totsize;
    Index ghostpts_o = num_ghost_points;

    //if (!myrank) fmt::print("[time={:.3f}] computed ghost points [treepts={},ghostpts={},pts_per_tree={:.1f},ghosts_per_tree={:.1f},distcomps={:.1f}M]\n", maxtime, totsize, num_ghost_points, totsize/(num_sites+0.0), num_ghost_points/(num_sites+0.0), distcomps/1000000.);
    #endif

    /*
     * Exchange points
     */

    IndexVector assignments(m), mysites, mytreeids, mytreeptrs, myghostids, myghostptrs;
    PointVector mytreepts, myghostpts;

    #ifdef STATS
    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();
    #endif

    for (Index i = 0; i < m; ++i)
    {
        assignments[i] = i % nprocs;
    }

    diagram.exchange_points(sendtreeids, sendtreeptrs, sendghostids, sendghostptrs, assignments, mysites, mytreeids, mytreeptrs, mytreepts);

    #ifdef STATS
    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    //if (!myrank) fmt::print("[time={:.3f}] exchanged and repacked points alltoall\n", maxtime);
    double exchange_points_time = maxtime;

    /*
     * Build ghost trees
     */
    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();
    #endif

    Index s = mysites.size();
    std::vector<GhostTree> ghost_trees(s);
    mydistcomps = 0;

    for (Index i = 0; i < s; ++i)
    {
        auto p1 = mytreepts.begin() + mytreeptrs[i];
        auto p2 = mytreepts.begin() + mytreeptrs[i+1];

        auto i1 = mytreeids.begin() + mytreeptrs[i];
        auto i2 = mytreeids.begin() + mytreeptrs[i+1];

        Index cellsize = diagram.get_cell_size(mysites[i]);
        ghost_trees[i].build(p1, p2, i1, i2, cellsize, mysites[i], cover, leaf_size, mydistcomps);
    }

    #ifdef STATS
    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    tot_distcomps += distcomps;

    //if (!myrank) fmt::print("[time={:.3f}] computed ghost trees [distcomps={:.1f}M]\n", maxtime, distcomps/1000000.);
    double compute_tree_time = maxtime;

    /*
     * Build epsilon graph
     */

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();
    #endif

    IndexVectorVector mygraph;
    IndexVector myids;

    mygraph.reserve(totsize/nprocs);
    myids.reserve(totsize/nprocs);

    Index my_n_edges = 0, n_edges;
    int done = 0;

    double t2 = 0, t3;
    double max_compute_time, sum_compute_time;
    mydistcomps = 0;

    do
    {
        const GhostTree *tree = ghost_trees.data();
        Index num_left = ghost_trees.size();

        t3 = -MPI_Wtime();
        for (Index i = 0; i < rebalance_rate && num_left > 0; ++i, ++tree, --num_left)
        {
            my_n_edges += tree->graph_query(mygraph, myids, epsilon, mydistcomps);
        }
        t3 += MPI_Wtime();
        t2 += t3;

        done = !!(num_left == 0);

        MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        if (!done)
        {
            std::vector<GhostTree> recv_trees;
            rebalance_trees(tree, num_left, recv_trees, MPI_COMM_WORLD);
            std::swap(ghost_trees, recv_trees);
        }

    } while (!done);

    #ifdef STATS
    t += MPI_Wtime();
    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t2, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t2, &sum_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    tottime += maxtime;

    MPI_Reduce(&my_n_edges, &n_edges, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&mydistcomps, &distcomps, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    tot_distcomps += distcomps;

    Real density = (n_edges+0.0)/totsize;


    //if (!myrank) fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},edges={},imbalance={:.3f},distcomps={:.1f}M]\n", maxtime, density, n_edges, nprocs*max_compute_time/sum_compute_time, distcomps/1000000.);
    //if (!myrank) fmt::print("[time={:.3f}] start-to-finish [qps={:.1f}K,distcomps={:.1f}M]\n", tottime, totsize/(tottime*1000.), tot_distcomps/1000000.);

    double compute_graph_time = maxtime;

    if (!myrank)
    {
        /*
         * filename
         * nprocs
         * cover
         * leaf
         * num_sites
         * random_sites
         * num_points
         * num_edges
         * density
         * voronoi_time
         * tree_points_time
         * exchange_points_time
         * compute_tree_time
         * compute_graph_time
         * distcomps
         */
        fmt::print("{}\t{}\t{:.2f}\t{}\t{}\t{}\t{}\t{}\t{:.2f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{}\n", points_fname, nprocs, cover, leaf_size, num_sites, random_sites, totsize, n_edges, density, voronoi_time, tree_points_time, exchange_points_time, compute_tree_time, compute_graph_time, tot_distcomps);
    }
    
    #endif

    #ifndef STATS
    t += MPI_Wtime();
    MPI_Reduce(&t, &tottime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&my_n_edges, &n_edges, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    Real density = (n_edges+0.0)/totsize;
    if (!myrank) fmt::print("[time={:.3f}] start-to-finish [qps={:.1f}K,density={:.3f},edges={}]\n", tottime, totsize/(tottime*1000.), density, n_edges);
    #endif

    if (graph_fname)
    {
        std::stringstream ss;

        if (!myrank)
        {
            ss << totsize << " " << n_edges << "\n";
        }

        for (Index k = 0; k < myids.size(); ++k)
        {
            Index i = myids[k];

            for (Index j : mygraph[k])
            {
                ss << i+1 << " " << j+1 << "\n";
            }
        }

        std::string s = ss.str();
        const char *mybuf = s.c_str();
        int count = s.size();

        MPI_Offset mysize = count;
        MPI_Offset fileoffset;

        MPI_Exscan(&mysize, &fileoffset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
        if (!myrank) fileoffset = 0;

        MPI_File fh;
        MPI_File_open(MPI_COMM_WORLD, graph_fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

        MPI_File_write_at_all(fh, fileoffset, mybuf, count, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fh);
    }

    MPI_Finalize();
    return 0;
}

void rebalance_trees(const GhostTree *send_trees, int sendcount, std::vector<GhostTree>& recv_trees, MPI_Comm comm)
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
