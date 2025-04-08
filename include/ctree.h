#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"
#include "fmt/ranges.h"

class CoverTree
{
    public:

        static inline constexpr Distance distance = Distance();

        struct Vertex
        {
            Index index;
            Point point;
            Real radius;
            Index child_ptr, num_children;
            Index leaf_ptr, num_leaves;
        };

        using VertexVector = std::vector<Vertex>;

        CoverTree() {}
        CoverTree(const PointVector& points) : points(points) {}

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

    private:

        PointVector points;
        VertexVector vertices;
        IndexVector children, leaves, ids;
        Index site;
};

#include "ctree.hpp"

#endif
