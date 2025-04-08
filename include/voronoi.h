#ifndef VORONOI_H_
#define VORONOI_H_

#include "utils.h"
#include "ctree.h"

class VoronoiDiagram
{
    public:

        static inline constexpr Distance distance = Distance();

        VoronoiDiagram(const Point *mypoints, Index mysize, Index myoffset, MPI_Comm comm);

        void build_random_diagram(Index m);
        void build_greedy_diagram(Index m);
        void build_replication_tree(Real cover, Index leaf_size);

        void find_ghost_neighbors(IndexVector& neighbors, Index query, Real epsilon) const;
        void compute_my_tree_points(IndexVector& mytreeids, IndexVector& mytreeptrs) const;
        Index compute_my_ghost_points(Real epsilon, IndexVector& myghostids, IndexVector& myghostptrs) const;
        void exchange_points(const IndexVector& sendtreeids, const IndexVector& sendtreeptrs, const IndexVector& sendghostids, const IndexVector& sendghostptrs, const IndexVector& assignments, IndexVector& mysites, IndexVector& mytreeids, IndexVector& mytreeptrs, PointVector& mytreepts) const;

        Index num_points() const { return mysize; }
        Index num_sites() const { return sites.size(); }

        Index get_cell_size(Index i) const { return cell_sizes[i]; }
        Index get_my_cell_size(Index i) const { return my_cell_sizes[i]; }
        Index get_farthest() const { return farthest; }
        Real get_radius() const { return radius; }

    private:

        const Point *mypoints;
        Index mysize, myoffset;
        MPI_Comm comm;

        IndexVector sites;
        PointVector site_points;

        IndexVector my_cell_sizes, cell_sizes;

        IndexVector mycells;
        RealVector mydists;

        Index farthest;
        Real radius;

        CoverTree reptree;

        struct Ball
        {
            Index id;
            Real radius;
            Point point;

            Ball() : id(0), radius(-1) {}
        };

        static void mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype);
        static void create_mpi_point(MPI_Datatype *MPI_POINT);
        static void create_mpi_ball(MPI_Datatype *MPI_BALL);
        static void create_mpi_argmax(MPI_Op *MPI_ARGMAX);
};

#include "voronoi.hpp"

#endif
