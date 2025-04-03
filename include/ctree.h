#ifndef COVER_TREE_H_
#define COVER_TREE_H_

#include "utils.h"

class CoverTree
{
    public:

        struct Vertex
        {
            Index id;
            Real radius;
            Index cptr, csize;
            Index lptr, lsize;

            Vertex() {}
            Vertex(Index id, Real radius) : id(id), radius(radius) {}
        };

        using VertexVector = std::vector<Vertex>;

        static inline constexpr Distance distance = Distance();

        CoverTree() {}
        CoverTree(const PointVector& points) : points(points), globids(points.size()) { std::iota(globids.begin(), globids.end(), (Index)0); }
        CoverTree(const Point *data, Index n) : points(data, data + n), globids(n) { std::iota(globids.begin(), globids.end(), (Index)0); }

        Index num_vertices() const { return vertices.size(); }
        Index num_points() const { return points.size(); }

        template <class PointIter>
        void assign_points(PointIter first, PointIter last);

        template <class PointIter, class IndexIter>
        void assign_points(PointIter pfirst, PointIter plast, IndexIter ifirst, IndexIter ilast);

        void build(Real covering_factor, Index leaf_size);
        void range_query(IndexVector& neighbors, const Point& query, Real radius) const;

        const Point *pdata() const { return points.data(); }
        const Index *idata() const { return globids.data(); }

        int get_packed_bufsize() const;
        int pack_tree(char *buf, MPI_Comm comm) const;
        void unpack_tree(const char *buf, int bufsize, MPI_Comm comm);

        void set_site(Index i) { site = i; }
        Index get_site() const { return site; }

    private:

        PointVector points;
        IndexVector globids;

        VertexVector vertices;
        IndexVector children, leaves;
        Index site;
};



#include "ctree.hpp"

#endif
