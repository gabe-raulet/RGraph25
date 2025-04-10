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

    Index num_graphs = 8;
    Real damping_factor = 0.5;
    Real sub_factor = -1;
    Index leaf_size = 50;
    Index rebalance_rate = 10;
    Real cover = 1.8;
    bool random_sites = false;

    PointVector mypoints;
    Index myoffset, totsize, mydistcomps = 0, distcomps;
    double t, maxtime, tottime = 0;

    json stats;
    json input, index_runtime;
    std::vector<json> graphs_runtime;

    const char *stats_fname = NULL;

    auto usage = [&] (int err, bool isroot)
    {
        if (isroot)
        {
            fprintf(stderr, "Usage: %s [options] <points> <epsilon> <num_sites>\n", argv[0]);
            fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", cover);
            fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
            fprintf(stderr, "         -n INT    rebalance rate [%d]\n", (int)rebalance_rate);
            fprintf(stderr, "         -N INT    number of graphs [%d]\n", (int)num_graphs);
            fprintf(stderr, "         -j FILE   json stats output\n");
            fprintf(stderr, "         -D FLOAT  damping factor [%.2f]\n", damping_factor);
            fprintf(stderr, "         -S FLOAT  sub factor (positive value supersedes -D) [%.2f]\n", sub_factor);
            fprintf(stderr, "         -R        choose sites randomly\n");
            fprintf(stderr, "         -h        help message\n");
        }

        MPI_Finalize();
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:j:Rn:N:D:S:h")) >= 0)
    {
        if      (c == 'c') cover = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'R') random_sites = true;
        else if (c == 'n') rebalance_rate = atoi(optarg);
        else if (c == 'N') num_graphs = atoi(optarg);
        else if (c == 'D') damping_factor = atof(optarg);
        else if (c == 'S') sub_factor = atof(optarg);
        else if (c == 'j') stats_fname = optarg;
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

    input["filename"] = points_fname;
    input["nprocs"] = nprocs;
    input["cover"] = cover;
    input["leaf"] = leaf_size;
    input["num_sites"] = num_sites;
    input["random_sites"] = random_sites;
    input["num_points"] = totsize;
    input["rebalance_rate"] = rebalance_rate;

    /*
     * Build Voronoi diagram
     */

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    VoronoiDiagram diagram(mypoints.data(), mypoints.size(), myoffset, MPI_COMM_WORLD);

    if (random_sites) diagram.build_random_diagram(m, mydistcomps);
    else diagram.build_greedy_diagram(m, mydistcomps);

    diagram.build_replication_tree(cover, leaf_size, mydistcomps);
    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={}]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest());

    index_runtime["sep"] = diagram.get_radius();
    index_runtime["farthest"] = diagram.get_farthest();

    /*
     * Compute tree points
     */
    Index num_ghost_points;
    IndexVector sendtreeids, sendtreeptrs, sendghostids, sendghostptrs;
    mydistcomps = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    diagram.compute_my_tree_points(sendtreeids, sendtreeptrs);
    num_ghost_points = diagram.compute_my_ghost_points(epsilon, sendghostids, sendghostptrs, mydistcomps);

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] computed ghost points [treepts={},ghostpts={},pts_per_tree={:.1f},ghosts_per_tree={:.1f}]\n", maxtime, totsize, num_ghost_points, totsize/(num_sites+0.0), num_ghost_points/(num_sites+0.0));
    index_runtime["ghostpts"] = num_ghost_points;

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

    t += MPI_Wtime();

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] computed ghost trees\n", maxtime);

    /*
     * Build epsilon graphs
     */

    for (Index iter = 0; iter < num_graphs; ++iter)
    {
        std::vector<GhostTree> trees = ghost_trees;
        graphs_runtime.emplace_back();
        json& graph_runtime = graphs_runtime.back();

        MPI_Barrier(MPI_COMM_WORLD);
        t = -MPI_Wtime();

        IndexVectorVector mygraph;
        IndexVector myids;

        Index my_n_edges = 0, n_edges;
        int done = 0;
        Index num_rebalances = 0;

        do
        {
            const GhostTree *tree = trees.data();
            Index num_left = trees.size();

            for (Index i = 0; i < rebalance_rate && num_left > 0; ++i, ++tree, --num_left)
            {
                my_n_edges += tree->graph_query(mygraph, myids, epsilon, mydistcomps);
            }

            done = !!(num_left == 0);

            MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

            if (!done)
            {
                std::vector<GhostTree> recv_trees;
                rebalance_trees(tree, num_left, recv_trees, MPI_COMM_WORLD);
                std::swap(trees, recv_trees);
                num_rebalances++;
            }

        } while (!done);

        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_n_edges, &n_edges, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        Real density = (n_edges+0.0)/totsize;

        if (!myrank) fmt::print("[time={:.3f}] built epsilon graph [epsilon={:.3f},density={:.3f},edges={}]\n", maxtime, epsilon, density, n_edges);

        graph_runtime["epsilon"] = epsilon;
        graph_runtime["num_edges"] = n_edges;
        graph_runtime["num_rebalances"] = num_rebalances;

        if (sub_factor < 0) epsilon *= damping_factor;
        else epsilon -= sub_factor;
    }

    stats["input"] = input;
    stats["index_runtime"] = index_runtime;
    stats["graphs_runtime"] = graphs_runtime;

    if (stats_fname && !myrank)
    {
        std::ofstream f(stats_fname);
        f << std::setw(4) << stats << std::endl;
        f.close();
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
