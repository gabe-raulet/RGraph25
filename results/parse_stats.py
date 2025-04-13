import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

#  exp = json.load(open("covtype.exp1.stats.n128.json", "r"))
exp = json.load(open("weak_scaling_results/stats.d5.a100.N32.json", "r"))

def read_rank_stats(exp):

    num_graphs = exp["stats"]["num_graphs"]
    nprocs = exp["stats"]["nprocs"]
    num_sites = exp["stats"]["num_sites"]
    random_sites = exp["stats"]["random_sites"]
    epsilons = exp["stats"]["epsilsons"]
    num_points = exp["stats"]["num_points"]
    filename = exp["stats"]["filename"]
    cover = exp["stats"]["cover"]
    leaf_size = exp["stats"]["leaf_size"]
    sep = exp["stats"]["sep"]

    # each row is a processor rank, each column is a graph
    distcomps = np.vstack([np.array(r["distcomps"]) for r in exp["rank_stats"]])
    num_edges = np.vstack([np.array(r["num_edges"]) for r in exp["rank_stats"]])
    query_times = np.vstack([np.array(r["query_times"]) for r in exp["rank_stats"]])

    # To sum across all ranks, use *.sum(axis=0)

    # each entry is a rank
    exchange_points_time = np.array([r["exchange_points_time"] for r in exp["rank_stats"]])
    ghost_points_time = np.array([r["ghost_points_time"] for r in exp["rank_stats"]])
    tree_build_time = np.array([r["tree_build_time"] for r in exp["rank_stats"]])
    voronoi_time = np.array([r["voronoi_time"] for r in exp["rank_stats"]])

    ghost_points_distcomps = np.array([r["ghost_points_distcomps"] for r in exp["rank_stats"]])
    tree_build_distcomps = np.array([r["tree_build_distcomps"] for r in exp["rank_stats"]])
    voronoi_distcomps = np.array([r["voronoi_distcomps"] for r in exp["rank_stats"]])

    num_ghost_points = np.array([r["num_ghost_points"] for r in exp["rank_stats"]])
    num_assigned_trees = np.array([r["num_assigned_trees"] for r in exp["rank_stats"]])
