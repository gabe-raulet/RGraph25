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

        Index num_vertices() const { return vertices.size(); }
        Index num_points() const { return points.size(); }

        void build(const PointVector& pts, Real cover, Index leaf_size);
        void range_query(IndexVector& neighbors, const Point& query, Real radius) const;

        void print_tree() const;

    private:

        PointVector points;
        VertexVector vertices;
        IndexVector children, leaves;
};

#include "ctree.hpp"

#endif
