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
#include <omp.h>
#include "utils.h"
#include "ctree.h"

Distance distance;

int main(int argc, char *argv[])
{
    const char *points_fname;
    const char *graph_fname = NULL;
    Real epsilon;

    Index leaf_size = 50;
    Real covering_factor = 1.8;
    bool verify_graph = false;
    int nthreads;

    PointVector points;
    double t, tottime = 0;

    auto usage = [&] (int err)
    {
        fprintf(stderr, "Usage: %s [options] <points> <epsilon>\n", argv[0]);
        fprintf(stderr, "Options: -c FLOAT  covering factor [%.2f]\n", covering_factor);
        fprintf(stderr, "         -l INT    leaf size [%lu]\n", (size_t)leaf_size);
        fprintf(stderr, "         -o FILE   graph output file\n");
        fprintf(stderr, "         -G        correctness check\n");
        fprintf(stderr, "         -h        help message\n");
        std::exit(err);
    };

    int c;
    while ((c = getopt(argc, argv, "c:l:o:Gh")) >= 0)
    {
        if      (c == 'c') covering_factor = atof(optarg);
        else if (c == 'l') leaf_size = atoi(optarg);
        else if (c == 'o') graph_fname = optarg;
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

    IndexVectorVector graph;
    CoverTree tree;

    read_fvecs(points, points_fname);

    t = -omp_get_wtime();
    tree.build(points, covering_factor, leaf_size);
    t += omp_get_wtime();
    tottime += t;

    /* tree.print_tree(); */

    fmt::print("[time={:.3f}] built cover tree [vertices={}]\n", t, tree.num_vertices());

    t = -omp_get_wtime();

    Index n = points.size();
    Index n_edges = 0;

    graph.resize(n);

    #pragma omp parallel for schedule(dynamic) reduction(+:n_edges)
    for (Index i = 0; i < n; ++i)
    {
        tree.range_query(graph[i], points[i], epsilon);
        n_edges += graph[i].size();
    }

    t += omp_get_wtime();
    tottime += t;

    Real density, sparsity;
    density = (n_edges+0.0)/n;
    sparsity = density/n;

    fmt::print("[time={:.3f}] built epsilon graph [density={:.3f},edges={},qps={:.3f}]\n", t, density, n_edges, n/t);
    fmt::print("[time={:.3f}] start-to-finish [qps={:.3f}]\n", tottime, n/tottime);

    if (verify_graph)
    {
        t = -omp_get_wtime();
        bool correct = check_correctness(points, graph, epsilon);
        t += omp_get_wtime();

        fmt::print("[time={:.3f}] {} correctness check [qps={:.3f}]\n", t, correct? "PASSED" : "FAILED", n/t);
    }

    if (graph_fname)
    {
        t = -omp_get_wtime();

        std::stringstream ss;
        ss << n << " " << n_edges << "\n";

        #pragma omp parallel
        {
            std::stringstream myss;

            #pragma omp for nowait
            for (Index i = 0; i < n; ++i)
            {
                for (Index j : graph[i])
                {
                    myss << (i+1) << " " << (j+1) << "\n";
                }
            }

            #pragma omp critical
            ss << myss.str();
        }

        std::string s = ss.str();
        const char *buf = s.c_str();

        FILE *f = fopen(graph_fname, "w");
        fprintf(f, "%s", buf);
        fclose(f);

        t += omp_get_wtime();

        fmt::print("[time={:.3f}] wrote graph to file '{}'\n", t, graph_fname);
    }

    return 0;
}
