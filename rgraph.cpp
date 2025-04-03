#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif

#include "json.hpp"
#include <fstream>
using json = nlohmann::json;

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
#include "ctree.h"

Distance distance;

int main(int argc, char *argv[])
{
    const char *points_fname;
    const char *graph_fname = NULL;
    const char *stats_fname = NULL;
    Real epsilon;

    Index leaf_size = 10;
    Real covering_factor = 1.3;
    bool verify_graph = false;
    bool header = false;
    int nthreads;

    double t, tottime = 0;
    PointVector points;
    json stats;

    auto usage = [&] (int err)
    {
        fprintf(stderr, "Usage: %s [options] <points> <epsilon>\n", argv[0]);
        fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
        fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
        fprintf(stderr, "         -o FILE   graph output file\n");
        fprintf(stderr, "         -j FILE   json stats file\n");
        fprintf(stderr, "         -H        include header\n");
        fprintf(stderr, "         -G        correctness check\n");
        fprintf(stderr, "         -h        help message\n");
        std::exit(err);
    };

    int c;

    while ((c = getopt(argc, argv, "c:l:j:o:HGh")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') graph_fname = optarg;
        else if (c == 'j') stats_fname = optarg;
        else if (c == 'H') header = true;
        else if (c == 'G') verify_graph = true;
        else if (c == 'h') usage(0);
    }

    if (argc - optind < 2)
    {
        fmt::print(stderr, "[err::{}] missing argument(s)\n", __func__);
        usage(1);
    }

    points_fname = argv[optind++];
    epsilon = atof(argv[optind]);

    #pragma omp parallel
    nthreads = omp_get_num_threads();

    /*
     * Read points
     */

    t = -omp_get_wtime();
    read_fvecs(points, points_fname);
    t += omp_get_wtime();

    #ifdef LOG
    fmt::print("[time={:.3f}] read {} points from file '{}'\n", t, points.size(), points_fname);
    #endif

    stats["filename"] = points_fname;
    stats["dimension"] = DIM_SIZE;
    stats["num_points"] = points.size();
    stats["epsilon"] = epsilon;

    /*
     * Build cover tree
     */

    t = -omp_get_wtime();

    CoverTree tree(points);
    tree.build(covering_factor, leaf_size);

    t += omp_get_wtime();
    tottime += t;

    #ifdef LOG
    fmt::print("[time={:.3f}] built cover tree [vertices={}]\n", t, tree.num_vertices());
    #endif

    stats["times"]["build_cover_tree"] = t;

    /*
     * Build epsilon graph
     */
    IndexVectorVector graph;

    t = -omp_get_wtime();

    Index n = points.size();
    graph.resize(n);

    Index n_edges = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:n_edges)
    for (Index i = 0; i < n; ++i)
    {
        tree.range_query(graph[i], points[i], epsilon);
        n_edges += graph[i].size();
    }

    t += omp_get_wtime();
    tottime += t;

    stats["times"]["build_epsilon_graph"] = t;

    Real density, sparsity;
    density = (n_edges+0.0)/n;
    sparsity = density / n;

    #ifdef LOG
    fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},sparsity={:.3f},edges={}]\n", t, density, sparsity, n_edges);
    fmt::print("[tottime={:.3f}] start-to-finish [queries_per_second={:.3f},edges_per_second={:.3f}]\n", tottime, n/tottime, n_edges/tottime);
    #endif

    #ifndef LOG
    if (header) fmt::print("prog\tfilename\tcover\tleaf\tnum_points\tthreads\ttime\tnum_edges\tdensity\tqueries_per_second\tedges_per_second\n");
    fmt::print("rgraph\t{}\t{:.2f}\t{}\t{}\t{}\t{:.3f}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\n", points_fname, covering_factor, leaf_size, n, nthreads, tottime, n_edges, density, n/tottime, n_edges/tottime);
    #endif

    stats["times"]["total"] = tottime;

    stats["num_edges"] = n_edges;
    stats["density"] = density;
    stats["sparsity"] = sparsity;
    stats["queries_per_second"] = n/tottime;
    stats["edges_per_second"] = n_edges/tottime;

    if (verify_graph)
    {
        /*
         * Check correctness
         */
        t = -omp_get_wtime();
        bool correct = check_correctness(points, graph, epsilon);
        t += omp_get_wtime();

        fmt::print("[time={:.3f}] {} correctness check\n", t, correct? "PASSED" : "FAILED");
    }

    if (graph_fname)
    {
        t = -omp_get_wtime();

        FILE *f = fopen(graph_fname, "w");

        fprintf(f, "%lld %lld\n", n, n_edges);

        for (Index i = 0; i < n; ++i)
        {
            std::sort(graph[i].begin(), graph[i].end());

            for (Index j : graph[i])
            {
                fprintf(f, "%lld %lld\n", i+1, j+1);
            }
        }

        fclose(f);

        t += omp_get_wtime();

        #ifdef LOG
        fmt::print("[time={:.3f}] wrote graph to file '{}'\n", t, graph_fname);
        #endif
    }

    if (stats_fname)
    {
        std::ofstream f(stats_fname);
        f << std::setw(4) << stats << std::endl;
        f.close();
    }

    return 0;
}
