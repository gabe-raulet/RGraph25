#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "fmt/ranges.h"

struct Vertex
{
    Index index;
    Point point;
    Real radius;
    Index child_ptr, num_children;
    Index leaf_ptr, num_leaves;
};

using VertexVector = std::vector<Vertex>;

class GhostTree;

class CoverTree
{
    public:

        friend class GhostTree;

        static inline constexpr Distance distance = Distance();

        CoverTree() {}

        Index num_vertices() const { return vertices.size(); }
        Index num_points() const { return leaf_points.size(); }

        void build(const PointVector& points, Real cover, Index leaf_size);

        Index range_query(IndexVector& neighbors, const Point& query, Real radius) const;

    private:

        PointVector leaf_points;
        VertexVector vertices;
        IndexVector children, leaves;
};

class GhostTree
{
    public:

        static inline constexpr Distance distance = Distance();

        GhostTree() {}
        GhostTree(const PointVector& points) : points(points) {}

        Index num_vertices() const { return vertices.size(); }
        Index num_points() const { return points.size(); }

        void build(Real cover, Index leaf_size);
        void build(const PointVector& pts, Real cover, Index leaf_size);

        template <class PointIter, class IndexIter>
        void build(PointIter pfirst, PointIter plast, IndexIter ifirst, IndexIter ilast, Real cover, Index leaf_size);

        Index range_query(IndexVector& neighbors, const Point& query, Real radius) const;
        Index graph_query(IndexVectorVector& graph, IndexVector& graphids, Index cellsize, Real radius) const;

        void print_tree() const;

        void set_site(Index i) { site = i; }
        Index get_site() const { return site; }

        int get_packed_bufsize() const;
        int pack_tree(char *buf, MPI_Comm comm) const;
        void unpack_tree(const char *buf, int bufsize, MPI_Comm comm);

    private:

        PointVector points;
        VertexVector vertices;
        IndexVector children, leaves, ids;
        Index site;
};

#include "ctree.hpp"

#endif
