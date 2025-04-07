VoronoiDiagram::VoronoiDiagram(const Point *mypoints, Index mysize, Index myoffset, MPI_Comm comm)
    : mypoints(mypoints),
      mysize(mysize),
      myoffset(myoffset),
      comm(comm),
      mycells(mysize),
      mydists(mysize, std::numeric_limits<Real>::max()) {}

void VoronoiDiagram::mpi_argmax(void *_in, void *_inout, int *len, MPI_Datatype *dtype)
{
    Ball *in = (Ball *)_in;
    Ball *inout = (Ball *)_inout;

    for (int i = 0; i < *len; ++i)
        if (in[i].radius > inout[i].radius)
            inout[i] = in[i];
}

void VoronoiDiagram::create_mpi_point(MPI_Datatype *MPI_POINT)
{
    MPI_Type_contiguous(DIM_SIZE, MPI_FLOAT, MPI_POINT);
    MPI_Type_commit(MPI_POINT);
}

void VoronoiDiagram::create_mpi_ball(MPI_Datatype *MPI_BALL)
{
    MPI_Datatype MPI_POINT;
    create_mpi_point(&MPI_POINT);
    int blklens[3] = {1,1,1};
    MPI_Aint disps[3] = {offsetof(Ball, id), offsetof(Ball, radius), offsetof(Ball, point)};
    MPI_Datatype types[3] = {MPI_INT64_T, MPI_FLOAT, MPI_POINT};
    MPI_Type_create_struct(3, blklens, disps, types, MPI_BALL);
    MPI_Type_commit(MPI_BALL);
    MPI_Type_free(&MPI_POINT);
}

void VoronoiDiagram::create_mpi_argmax(MPI_Op *MPI_ARGMAX)
{
    MPI_Op_create(&mpi_argmax, 1, MPI_ARGMAX);
}

void VoronoiDiagram::build_random_diagram(Index m)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_POINT, MPI_BALL;
    MPI_Op MPI_ARGMAX;

    create_mpi_point(&MPI_POINT);
    create_mpi_ball(&MPI_BALL);
    create_mpi_argmax(&MPI_ARGMAX);

    sites.resize(m);
    site_points.resize(m);
    cell_sizes.resize(m);
    my_cell_sizes.resize(m, 0);

    int sendcount;
    std::vector<int> recvcounts(nprocs), rdispls(nprocs);

    if (myoffset < m) sendcount = std::min((int)mysize, (int)(m - myoffset));
    else sendcount = 0;

    MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    MPI_Allgatherv(mypoints, sendcount, MPI_POINT, site_points.data(), recvcounts.data(), rdispls.data(), MPI_POINT, comm);

    for (Index i = 0; i < m; ++i)
        sites[i] = i;

    for (Index p = 0; p < mysize; ++p)
    {
        for (Index i = 0; i < m; ++i)
        {
            Real d = distance(site_points[i], mypoints[p]);

            if (d < mydists[p])
            {
                mydists[p] = d;
                mycells[p] = i;
            }
        }
    }

    Ball ball;

    ball.id = (std::max_element(mydists.begin(), mydists.end()) - mydists.begin()) + myoffset;
    ball.radius = mydists[ball.id-myoffset];
    ball.point = mypoints[ball.id-myoffset];

    MPI_Allreduce(MPI_IN_PLACE, &ball, 1, MPI_BALL, MPI_ARGMAX, comm);

    farthest = ball.id;
    radius = ball.radius;

    for (Index p = 0; p < mysize; ++p)
        my_cell_sizes[mycells[p]]++;

    MPI_Allreduce(my_cell_sizes.data(), cell_sizes.data(), static_cast<int>(m), MPI_INT64_T, MPI_SUM, comm);

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_BALL);
    MPI_Op_free(&MPI_ARGMAX);
}

void VoronoiDiagram::build_greedy_diagram(Index m)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    MPI_Datatype MPI_POINT, MPI_BALL;
    MPI_Op MPI_ARGMAX;

    create_mpi_point(&MPI_POINT);
    create_mpi_ball(&MPI_BALL);
    create_mpi_argmax(&MPI_ARGMAX);

    sites.resize(m);
    site_points.resize(m);
    cell_sizes.resize(m);
    my_cell_sizes.resize(m, 0);

    Ball ball;

    ball.id = 0;
    ball.point = mypoints[0];

    MPI_Bcast(&ball.point, 1, MPI_POINT, 0, comm);

    /*
     * Pick m sites using greedy permutation selection
     */
    for (Index i = 0; i < m; ++i)
    {
        sites[i] = ball.id;
        site_points[i] = ball.point;

        ball.radius = 0;
        ball.id = 0;

        /*
         * In parallel, compute distance of each point to newest site
         * and update cell pointers if a point switches Voronoi cells.
         *
         * Simultaneously, keep track of which point is the farthest
         * from its Voronoi site (it may have changed)
         */

        for (Index p = 0; p < mysize; ++p)
        {
            Real d = distance(mypoints[p], site_points[i]);

            if (d < mydists[p])
            {
                mycells[p] = i;
                mydists[p] = d;
            }

            if (mydists[p] > ball.radius)
            {
                ball.id = myoffset + p;
                ball.radius = mydists[p];
            }
        }

        ball.point = mypoints[ball.id-myoffset];

        /*
         * Every processor contributes its local farthest found point in an allreduce
         */
        MPI_Allreduce(MPI_IN_PLACE, &ball, 1, MPI_BALL, MPI_ARGMAX, comm);
    }

    farthest = ball.id;
    radius = ball.radius;

    for (Index p = 0; p < mysize; ++p)
    {
        Index cell = mycells[p];
        my_cell_sizes[cell]++;
    }

    MPI_Allreduce(my_cell_sizes.data(), cell_sizes.data(), static_cast<int>(m), MPI_INT64_T, MPI_SUM, comm);

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_BALL);
    MPI_Op_free(&MPI_ARGMAX);
}

