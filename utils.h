#ifndef UTILS_H_
#define UTILS_H_

#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <assert.h>
#include <omp.h>
#include <mpi.h>

#ifndef DIM_SIZE
#error "DIM_SIZE is undefined"
#endif

using Real = float;
using Index = int64_t;
using Point = std::array<Real, DIM_SIZE>;
using PointRecord = std::array<char, sizeof(int) + sizeof(Point)>;

using RealVector = std::vector<Real>;
using IndexVector = std::vector<Index>;
using PointVector = std::vector<Point>;

using IndexMap = std::unordered_map<Index, Index>;

using IndexVectorVector = std::vector<IndexVector>;
using PointVectorVector = std::vector<PointVector>;

struct Distance
{
    Real operator()(const Point& p, const Point& q) const;
};

template <class Integer>
void get_balanced_counts(std::vector<Integer>& counts, size_t totsize);

void read_fvecs(PointVector& points, const char *fname);
void read_fvecs(PointVector& mypoints, Index& myoffset, Index& totsize, const char *fname, MPI_Comm comm);

void write_fvecs(const PointVector& points, const char *fname);

bool check_correctness(const PointVector& points, IndexVectorVector& graph, Real epsilon);

#include "utils.hpp"

#endif
