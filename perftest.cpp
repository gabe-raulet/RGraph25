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

using DoubleVector = std::vector<double>;

struct StatsCollector
{
    /* input stats */
    const char *filename;
    int nprocs;
    Real cover;
    Index leaf_size;
    Index num_sites;
    bool random_sites;
    Index num_points;
    Index num_graphs;
    Index rebalance_rate;

    /* index_runtime stats */
    Real sep;
    Index farthest;

    Index my_num_ghost_points;
    Index my_num_assigned_trees;
    Index my_voronoi_distcomps;
    Index my_ghost_points_distcomps;
    Index my_tree_build_distcomps;

    double my_voronoi_time;
    double my_ghost_points_time;
    double my_exchange_points_time;
    double my_tree_build_time;

    DoubleVector epsilons;
    IndexVector num_rebalances;

    IndexVector my_num_edges;
    IndexVector my_distcomps;
    DoubleVector my_query_times;
    DoubleVector my_rebalance_times;

    void write_json(const char *json_fname, MPI_Comm comm);

    void push_graph_stats(double epsilon, Index num_rebalances_count, Index my_n_edges, double my_query_time, double my_rebalance_time, Index mydistcomps)
    {
        epsilons.push_back(epsilon);
        num_rebalances.push_back(num_rebalances_count);
        my_num_edges.push_back(my_n_edges);
        my_query_times.push_back(my_query_time);
        my_rebalance_times.push_back(my_rebalance_time);
        my_distcomps.push_back(mydistcomps);
    }
};

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

    /* json stats; */
    /* json input, index_runtime; */
    /* std::vector<json> graphs_runtime; */

    StatsCollector stats;

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

    stats.filename = points_fname;
    stats.num_graphs = num_graphs;
    stats.nprocs = nprocs;
    stats.cover = cover;
    stats.leaf_size = leaf_size;
    stats.num_sites = num_sites;
    stats.random_sites = random_sites;
    stats.num_points = totsize;
    stats.rebalance_rate = rebalance_rate;

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
    stats.my_voronoi_time = t;
    stats.my_voronoi_distcomps = mydistcomps;

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!myrank) fmt::print("[time={:.3f}] built r-net Voronoi diagram [sep={:.3f},num_sites={},farthest={}]\n", maxtime, diagram.get_radius(), diagram.num_sites(), diagram.get_farthest());

    stats.sep = diagram.get_radius();
    stats.farthest = diagram.get_farthest();

    /*
     * Compute tree points
     */
    Index num_ghost_points;
    IndexVector sendtreeids, sendtreeptrs, sendghostids, sendghostptrs;
    mydistcomps = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    t = -MPI_Wtime();

    diagram.compute_my_tree_points(sendtreeids, sendtreeptrs);
    stats.my_num_ghost_points = diagram.compute_my_ghost_points(epsilon, sendghostids, sendghostptrs, mydistcomps);

    t += MPI_Wtime();
    stats.my_ghost_points_time = t;
    stats.my_ghost_points_distcomps = mydistcomps;

    MPI_Reduce(&t, &maxtime, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    MPI_Reduce(&stats.my_num_ghost_points, &num_ghost_points, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

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
    stats.my_exchange_points_time = t;

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
    stats.my_tree_build_time = t;
    stats.my_tree_build_distcomps = mydistcomps;

    MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    tottime += maxtime;

    if (!myrank) fmt::print("[time={:.3f}] computed ghost trees\n", maxtime);
    stats.my_num_assigned_trees = s;

    /*
     * Build epsilon graphs
     */

    for (Index iter = 0; iter < num_graphs; ++iter)
    {
        std::vector<GhostTree> trees = ghost_trees;
        /* graphs_runtime.emplace_back(); */
        /* json& graph_runtime = graphs_runtime.back(); */

        MPI_Barrier(MPI_COMM_WORLD);
        t = -MPI_Wtime();

        IndexVectorVector mygraph;
        IndexVector myids;

        Index my_n_edges = 0, n_edges;
        int done = 0;

        Index num_rebalances = 0;
        double my_query_time = 0;
        double my_rebalance_time = 0;
        mydistcomps = 0;

        double t2;

        do
        {
            t2 = -MPI_Wtime();

            const GhostTree *tree = trees.data();
            Index num_left = trees.size();

            for (Index i = 0; i < rebalance_rate && num_left > 0; ++i, ++tree, --num_left)
            {
                my_n_edges += tree->graph_query(mygraph, myids, epsilon, mydistcomps);
            }

            done = !!(num_left == 0);

            t2 += MPI_Wtime();
            my_query_time += t2;

            MPI_Allreduce(MPI_IN_PLACE, &done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

            t2 = -MPI_Wtime();

            if (!done)
            {
                std::vector<GhostTree> recv_trees;
                rebalance_trees(tree, num_left, recv_trees, MPI_COMM_WORLD);
                std::swap(trees, recv_trees);
                num_rebalances++;
            }

            t2 += MPI_Wtime();
            my_rebalance_time += t2;

        } while (!done);

        t += MPI_Wtime();

        MPI_Reduce(&t, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&my_n_edges, &n_edges, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

        Real density = (n_edges+0.0)/totsize;

        if (!myrank) fmt::print("[time={:.3f}] built epsilon graph [epsilon={:.3f},density={:.3f},edges={}]\n", maxtime, epsilon, density, n_edges);

        /* graph_runtime["epsilon"] = epsilon; */
        /* graph_runtime["num_edges"] = n_edges; */
        /* graph_runtime["num_rebalances"] = num_rebalances; */

        stats.push_graph_stats(epsilon, num_rebalances, my_n_edges, my_query_time, my_rebalance_time, mydistcomps);

        if (sub_factor < 0) epsilon *= damping_factor;
        else epsilon -= sub_factor;
    }

    /* stats["input"] = input; */
    /* stats["index_runtime"] = index_runtime; */
    /* stats["graphs_runtime"] = graphs_runtime; */

    stats.write_json(stats_fname, MPI_COMM_WORLD);

    /* if (stats_fname && !myrank) */
    /* { */
        /* std::ofstream f(stats_fname); */
        /* f << std::setw(4) << stats << std::endl; */
        /* f.close(); */
    /* } */


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

void StatsCollector::write_json(const char *json_fname, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    struct RankStats
    {
        Index num_ghost_points;
        Index num_assigned_trees;
        Index voronoi_distcomps;
        Index ghost_points_distcomps;
        Index tree_build_distcomps;

        double voronoi_time;
        double ghost_points_time;
        double exchange_points_time;
        double tree_build_time;

        /* IndexVector num_edges; */
        /* IndexVector distcomps; */

        /* DoubleVector query_times; */
        /* DoubleVector rebalance_times; */

        json get_json() const
        {
            json obj;
            obj["num_ghost_points"] = num_ghost_points;
            obj["num_assigned_trees"] = num_assigned_trees;
            obj["voronoi_distcomps"] = voronoi_distcomps;
            obj["ghost_points_distcomps"] = ghost_points_distcomps;
            obj["tree_build_distcomps"] = tree_build_distcomps;
            obj["voronoi_time"] = voronoi_time;
            obj["ghost_points_time"] = ghost_points_time;
            obj["exchange_points_time"] = exchange_points_time;
            obj["tree_build_time"] = tree_build_time;

            /* obj["my_num_edges"] = num_edges; */
            /* obj["my_distcomps"] = distcomps; */

            /* obj["my_query_times"] = query_times; */
            /* obj["my_rebalance_times"] = rebalance_times; */

            return obj;
        }
    };

    MPI_Datatype MPI_RANK_STATS;

    int blklens[2] = {5,4};
    MPI_Aint disps[2] = {offsetof(RankStats, num_ghost_points), offsetof(RankStats, voronoi_time)};
    MPI_Datatype types[2] = {MPI_INT64_T, MPI_DOUBLE};
    MPI_Type_create_struct(2, blklens, disps, types, &MPI_RANK_STATS);
    MPI_Type_commit(&MPI_RANK_STATS);

    RankStats my_rank_stats;
    std::vector<RankStats> rank_stats;

    my_rank_stats.num_ghost_points = my_num_ghost_points;
    my_rank_stats.num_assigned_trees = my_num_assigned_trees;
    my_rank_stats.voronoi_distcomps = my_voronoi_distcomps;
    my_rank_stats.ghost_points_distcomps = my_ghost_points_distcomps;
    my_rank_stats.tree_build_distcomps = my_tree_build_distcomps;
    my_rank_stats.voronoi_time = my_voronoi_time;
    my_rank_stats.ghost_points_time = my_ghost_points_time;
    my_rank_stats.exchange_points_time = my_exchange_points_time;
    my_rank_stats.tree_build_time = my_tree_build_time;

    /* my_rank_stats.num_edges = my_num_edges; */
    /* my_rank_stats.distcomps = my_distcomps; */
    /* my_rank_stats.query_times = my_query_times; */
    /* my_rank_stats.rebalance_times = my_rebalance_times; */

    IndexVector num_edges, distcomps;
    DoubleVector query_times, rebalance_times;

    if (!myrank)
    {
        rank_stats.resize(nprocs);

        num_edges.resize(nprocs*num_graphs);
        distcomps.resize(nprocs*num_graphs);
        query_times.resize(nprocs*num_graphs);
        rebalance_times.resize(nprocs*num_graphs);
    }

    MPI_Gather(&my_rank_stats, 1, MPI_RANK_STATS, rank_stats.data(), 1, MPI_RANK_STATS, 0, comm);

    MPI_Gather(my_num_edges.data(), static_cast<int>(num_graphs), MPI_INT64_T, num_edges.data(), static_cast<int>(num_graphs), MPI_INT64_T, 0, comm);
    MPI_Gather(my_distcomps.data(), static_cast<int>(num_graphs), MPI_INT64_T, distcomps.data(), static_cast<int>(num_graphs), MPI_INT64_T, 0, comm);
    MPI_Gather(my_query_times.data(), static_cast<int>(num_graphs), MPI_DOUBLE, query_times.data(), static_cast<int>(num_graphs), MPI_DOUBLE, 0, comm);
    MPI_Gather(my_rebalance_times.data(), static_cast<int>(num_graphs), MPI_DOUBLE, rebalance_times.data(), static_cast<int>(num_graphs), MPI_DOUBLE, 0, comm);

    if (!myrank)
    {
        json stats_json;
        stats_json["filename"] = filename;
        stats_json["num_graphs"] = num_graphs;
        stats_json["nprocs"] = nprocs;
        stats_json["cover"] = cover;
        stats_json["leaf_size"] = leaf_size;
        stats_json["num_sites"] = num_sites;
        stats_json["random_sites"] = random_sites;
        stats_json["num_points"] = num_points;
        stats_json["rebalance_rate"] = rebalance_rate;
        stats_json["sep"] = sep;
        stats_json["farthest"] = farthest;
        stats_json["epsilsons"] = epsilons;
        stats_json["num_rebalances"] = num_rebalances;

        std::vector<json> rank_stats_json;

        for (const RankStats& o : rank_stats)
        {
            rank_stats_json.push_back(o.get_json());
        }

            /* obj["my_num_edges"] = num_edges; */
            /* obj["my_distcomps"] = distcomps; */

            /* obj["my_query_times"] = query_times; */
            /* obj["my_rebalance_times"] = rebalance_times; */

        for (Index i = 0; i < nprocs; ++i)
        {
            rank_stats_json[i]["num_edges"] = std::vector(num_edges.begin() + i*num_graphs, num_edges.begin() + (i+1)*num_graphs);
            rank_stats_json[i]["distcomps"] = std::vector(distcomps.begin() + i*num_graphs, distcomps.begin() + (i+1)*num_graphs);
            rank_stats_json[i]["query_times"] = std::vector(query_times.begin() + i*num_graphs, query_times.begin() + (i+1)*num_graphs);
            rank_stats_json[i]["rebalance_times"] = std::vector(rebalance_times.begin() + i*num_graphs, rebalance_times.begin() + (i+1)*num_graphs);
        }

        json result;
        result["stats"] = stats_json;
        result["rank_stats"] = rank_stats_json;

        std::ofstream f(json_fname);
        f << std::setw(4) << result << std::endl;
        f.close();
    }



    ///* input stats */
    //const char *filename;
    //int nprocs;
    //Real cover;
    //Index leaf_size;
    //Index num_sites;
    //bool random_sites;
    //Index num_points;
    //Index rebalance_rate;

    ///* index_runtime stats */
    //Real sep;
    //Index farthest;
    //Index my_num_ghost_points;
    //Index my_num_assigned_trees;

    //Real my_voronoi_time;
    //Real my_ghost_points_time;
    //Real my_exchange_points_time;
    //Real my_tree_build_time;


    /* if (!myrank) */
    /* { */
        /* std::ofstream f(stats_fname); */
        /* f << std::setw(4) << stats << std::endl; */
        /* f.close(); */
    /* } */

    MPI_Type_free(&MPI_RANK_STATS);
}
