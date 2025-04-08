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

        Index num_vertices() const { return tree.num_vertices(); }
        Index num_points() const { return tree.num_points(); }

        template <class PointIter, class IndexIter>
        void build(PointIter p1, PointIter p2, IndexIter i1, IndexIter i2, Index cellsize, Index site, Real cover, Index leaf_size);

        Index graph_query(IndexVectorVector& graph, IndexVector& graphids, Real radius) const;

        int get_packed_bufsize() const;
        int pack_tree(char *buf, MPI_Comm comm) const;
        void unpack_tree(const char *buf, int bufsize, MPI_Comm comm);

        Index get_site() const { return site; }

    private:

        CoverTree tree;
        PointVector points;
        IndexVector ids;
        Index site;
};

#include "ctree.hpp"

#endif
