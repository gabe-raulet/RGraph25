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
    const char *graph_fname = NULL;
    const char *stats_fname = NULL;

    Real epsilon;
    Index num_sites;

    Index leaf_size = 10;
    Real covering_factor = 1.3;
    Index rebalance_rate = 10;
    bool verify_graph = false;
    bool random_diagram = false;
    bool header = false;
    int nthreads;

    double t, maxtime, tottime = 0;
    PointVector mypoints;
    Index myoffset, totsize;

    json stats;

    auto usage = [&] (int err, bool isroot)
    {
        if (isroot)
        {
            fprintf(stderr, "Usage: %s [options] <points> <epsilon> <num_sites>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
            fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
            fprintf(stderr, "         -n INT    rebalance rate [%lld]\n", rebalance_rate);
            fprintf(stderr, "         -o FILE   graph output file\n");
            fprintf(stderr, "         -j FILE   json stats file\n");
            fprintf(stderr, "         -R        choose sites randomly\n");
            fprintf(stderr, "         -H        include header\n");
            fprintf(stderr, "         -G        correctness check\n");
            fprintf(stderr, "         -h        help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;

    while ((c = getopt(argc, argv, "c:l:j:o:n:RHGh")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') graph_fname = optarg;
        else if (c == 'j') stats_fname = optarg;
        else if (c == 'n') rebalance_rate = atoi(optarg);
        else if (c == 'R') random_diagram = true;
        else if (c == 'H') header = true;
        else if (c == 'G') verify_graph = true;
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

    /*
     * Read points
     */

    MPI_Barrier(MPI_COMM_WORLD);

    t = -MPI_Wtime();
    read_fvecs(mypoints, myoffset, totsize, points_fname, MPI_COMM_WORLD);
    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] read {} points from file '{}'\n", maxtime, totsize, points_fname);
    #endif

    assert((myoffset == myrank*(totsize/nprocs)));

    Index n = mypoints.size();
    Index m = num_sites;

    stats["filename"] = points_fname;
    stats["dimension"] = DIM_SIZE;
    stats["num_points"] = n;
    stats["num_sites"] = m;
    stats["epsilon"] = epsilon;

    /*
     * Build Voronoi diagram
     */

    t = -MPI_Wtime();

    VoronoiDiagram diagram(mypoints.data(), mypoints.size(), myoffset, MPI_COMM_WORLD);

    if (random_diagram) diagram.build_random_diagram(m);
    else diagram.build_greedy_permutation(m);

    diagram.build_replication_tree(covering_factor, leaf_size);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={}]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest());
    #endif

    stats["times"]["build_voronoi_diagram"] = maxtime;

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

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] computed ghost points [treepts={},ghostpts={},pts_per_tree={:.1f},ghosts_per_tree={:.1f}]\n", maxtime, totsize, num_ghost_points, totsize/(num_sites+0.0), num_ghost_points/(num_sites+0.0));
    #endif

    stats["times"]["compute_tree_points"] = maxtime;

    /*
     * Exchange points
     */

    MPI_Barrier(MPI_COMM_WORLD);

    IndexVector assignments(m), mysites, mytreeids, mytreeptrs, myghostids, myghostptrs;
    PointVector mytreepts, myghostpts;

    t = -MPI_Wtime();

    for (Index i = 0; i < m; ++i)
    {
        assignments[i] = i % nprocs;
    }

    diagram.exchange_points(sendtreeids, sendtreeptrs, sendghostids, sendghostptrs, assignments, mysites, mytreeids, mytreeptrs, mytreepts);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] exchanged and repacked points alltoall\n", maxtime);
    #endif

    stats["times"]["exchange_points"] = maxtime;

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

        ghost_trees[i].assign_points(pfirst, plast, ifirst, ilast);
        ghost_trees[i].build(covering_factor, leaf_size);
        ghost_trees[i].set_site(mysites[i]);
    }

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] computed ghost trees\n", maxtime);
    #endif

    stats["times"]["build_trees"] = maxtime;

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
            const Point *pts = tree->pdata();
            const Index *ids = tree->idata();

            Index cellsize = diagram.get_cell_size(tree->get_site());

            for (Index j = 0; j < cellsize; ++j)
            {
                mygraph.emplace_back();
                myids.push_back(ids[j]);
                tree->range_query(mygraph.back(), pts[j], epsilon);
                my_n_edges += mygraph.back().size();
            }
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

    #ifdef LOG
    if (!myrank) fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},sparsity={:.3f},edges={}]\n", maxtime, density, sparsity, n_edges);
    if (!myrank) fmt::print("[tottime={:.3f}] start-to-finish [queries_per_second={:.3f},edges_per_second={:.3f}]\n", tottime, totsize/tottime, n_edges/tottime);
    #endif

    #ifndef LOG
    if (header && !myrank) fmt::print("prog\tfilename\tcover\tleaf\tnum_points\tranks\ttime\tnum_edges\tdensity\tqueries_per_second\tedges_per_second\tnum_sites\trandom_sites\n");
    if (!myrank) fmt::print("rgraph_mpi\t{}\t{:.2f}\t{}\t{}\t{}\t{:.3f}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{}\t{}\n", points_fname, covering_factor, leaf_size, totsize, nprocs, tottime, n_edges, density, totsize/tottime, n_edges/tottime, num_sites, random_diagram);
    #endif

    if (stats_fname && !myrank)
    {
        std::ofstream f(stats_fname);
        f << std::setw(4) << stats << std::endl;
        f.close();
    }

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

void rebalance_trees(const CoverTree *send_trees, int sendcount, std::vector<CoverTree>& recv_trees, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    /* std::vector<int> counts; */
    /* if (!myrank) counts.resize(nprocs); */

    /* MPI_Gather(&sendcount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm); */

    /* if (!myrank) */
    /* { */
        /* fmt::print("Rebalancing: {}\n", counts); */
    /* } */

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

    /* MPI_Gather(&recvcount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm); */

    /* if (!myrank) */
    /* { */
        /* fmt::print("Rebalanced:  {}\n", counts); */
    /* } */
}
