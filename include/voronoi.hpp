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

    #pragma omp parallel for
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

    #pragma omp parallel for
    for (Index p = 0; p < mysize; ++p)
    {
        #pragma omp atomic update
        my_cell_sizes[mycells[p]]++;
    }

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

    #pragma omp declare reduction(argmax_reduce : Ball : \
            omp_out = (omp_out.radius > omp_in.radius)? omp_out : omp_in) \
            initializer(omp_priv = Ball())


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

        #pragma omp parallel for reduction(argmax_reduce:ball)
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

    #pragma omp parallel for
    for (Index p = 0; p < mysize; ++p)
    {
        Index cell = mycells[p];

        #pragma omp atomic update
        my_cell_sizes[cell]++;
    }

    MPI_Allreduce(my_cell_sizes.data(), cell_sizes.data(), static_cast<int>(m), MPI_INT64_T, MPI_SUM, comm);

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_BALL);
    MPI_Op_free(&MPI_ARGMAX);
}

void VoronoiDiagram::build_replication_tree(Real covering_factor, Index leaf_size)
{
    reptree.build(site_points, covering_factor, leaf_size);
}

void VoronoiDiagram::find_ghost_neighbors(IndexVector& neighbors, Index query, Real epsilon) const
{
    reptree.range_query(neighbors, mypoints[query], mydists[query] + 2*epsilon);

    auto it = std::remove_if(neighbors.begin(), neighbors.end(), [&](Index id) { return id == mycells[query]; });
    neighbors.erase(it, neighbors.end());
}

void VoronoiDiagram::compute_my_tree_points(IndexVector& mytreeids, IndexVector& mytreeptrs) const
{
    Index m = num_sites();

    mytreeptrs.resize(m);
    std::exclusive_scan(my_cell_sizes.begin(), my_cell_sizes.end(), mytreeptrs.begin(), static_cast<Index>(0));
    mytreeptrs.push_back(my_cell_sizes.back() + mytreeptrs.back());
    mytreeids.resize(mytreeptrs.back());

    IndexVector ptrs = mytreeptrs;

    for (Index p = 0; p < mysize; ++p)
    {
        mytreeids[ptrs[mycells[p]]++] = p + myoffset;
    }
}

Index VoronoiDiagram::compute_my_ghost_points(Real epsilon, IndexVector& myghostids, IndexVector& myghostptrs) const
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index m = num_sites();
    IndexVectorVector myneighbors(mysize);
    IndexVector myghostcounts(m, 0);

    Index my_num_ghost_points = 0, num_ghost_points;

    Index *treecounts = myghostcounts.data();

    #pragma omp parallel for schedule(dynamic) reduction(+:treecounts[:m]) reduction(+:my_num_ghost_points)
    for (Index p = 0; p < mysize; ++p)
    {
        find_ghost_neighbors(myneighbors[p], p, epsilon);

        for (Index j : myneighbors[p])
            treecounts[j]++;

        my_num_ghost_points += myneighbors[p].size();
    }

    myghostptrs.resize(m);
    std::exclusive_scan(myghostcounts.begin(), myghostcounts.end(), myghostptrs.begin(), static_cast<Index>(0));
    myghostptrs.push_back(myghostcounts.back() + myghostptrs.back());

    myghostids.resize(myghostptrs.back());
    IndexVector ptrs = myghostptrs;

    for (Index p = 0; p < mysize; ++p)
        for (Index j : myneighbors[p])
            myghostids[ptrs[j]++] = p + myoffset;

    MPI_Allreduce(&my_num_ghost_points, &num_ghost_points, 1, MPI_INT64_T, MPI_SUM, comm);

    return num_ghost_points;
}

void VoronoiDiagram::exchange_points(const IndexVector& sendtreeids, const IndexVector& sendtreeptrs, const IndexVector& sendghostids, const IndexVector& sendghostptrs, const IndexVector& assignments, IndexVector& mysites, IndexVector& mytreeids, IndexVector& mytreeptrs, PointVector& mytreepts) const
{
    mysites.clear();
    mytreeids.clear();
    mytreeptrs.clear();
    mytreepts.clear();

    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    Index m = num_sites();

    struct PointEnvelope
    {
        Point point;
        Index id;
        Index cell;
        int ghost;

        PointEnvelope() {}
        PointEnvelope(Point point, Index id, Index cell, int ghost) : point(point), id(id), cell(cell), ghost(ghost) {}
    };

    using PointEnvelopeVector = std::vector<PointEnvelope>;

    MPI_Datatype MPI_POINT, MPI_POINT_ENVELOPE;

    create_mpi_point(&MPI_POINT);

    int blklens[3] = {1,2,1};
    MPI_Aint disps[3] = {offsetof(PointEnvelope, point), offsetof(PointEnvelope, id), offsetof(PointEnvelope, ghost)};
    MPI_Datatype types[3] = {MPI_POINT, MPI_INT64_T, MPI_INT};
    MPI_Type_create_struct(3, blklens, disps, types, &MPI_POINT_ENVELOPE);
    MPI_Type_commit(&MPI_POINT_ENVELOPE);

    std::vector<PointEnvelopeVector> sendbufs(nprocs);
    PointEnvelopeVector sendbuf, recvbuf;
    IndexMap myslots;

    for (Index i = 0; i < m; ++i)
    {
        int dest = assignments[i];

        if (dest == myrank)
        {
            myslots.insert({i, mysites.size()});
            mysites.push_back(i);
        }

        for (Index j = sendtreeptrs[i]; j < sendtreeptrs[i+1]; ++j)
        {
            Index id = sendtreeids[j];
            Point point = mypoints[id-myoffset];
            sendbufs[dest].emplace_back(point, id, i, 0);
        }

        for (Index j = sendghostptrs[i]; j < sendghostptrs[i+1]; ++j)
        {
            Index id = sendghostids[j];
            Point point = mypoints[id-myoffset];
            sendbufs[dest].emplace_back(point, id, i, 1);
        }
    }

    std::vector<int> sendcounts(nprocs), recvcounts(nprocs), sdispls(nprocs), rdispls(nprocs);

    for (int i = 0; i < nprocs; ++i)
    {
        sdispls[i] = sendbuf.size();
        sendcounts[i] = sendbufs[i].size();
        sendbuf.insert(sendbuf.end(), sendbufs[i].begin(), sendbufs[i].end());
    }

    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), static_cast<int>(0));
    recvbuf.resize(recvcounts.back() + rdispls.back());

    MPI_Alltoallv(sendbuf.data(), sendcounts.data(), sdispls.data(), MPI_POINT_ENVELOPE,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_POINT_ENVELOPE, comm);

    std::sort(recvbuf.begin(), recvbuf.end(), [](const auto& lhs, const auto& rhs) { return lhs.ghost < rhs.ghost; });

    Index s = mysites.size();
    IndexVector myrecvcounts(s);

    Index numrecv = recvbuf.size();

    #pragma omp parallel for
    for (Index i = 0; i < numrecv; ++i)
    {
        Index slot = myslots.find(recvbuf[i].cell)->second;

        #pragma omp atomic update
        myrecvcounts[slot]++;
    }

    /* for (const auto& [point, id, cell, ghost] : recvbuf) */
    /* { */
        /* Index slot = myslots.find(cell)->second; */
        /* myrecvcounts[slot]++; */
    /* } */

    mytreeptrs.resize(s);
    std::exclusive_scan(myrecvcounts.begin(), myrecvcounts.end(), mytreeptrs.begin(), static_cast<Index>(0));
    mytreeptrs.push_back(mytreeptrs.back() + myrecvcounts.back());
    mytreepts.resize(mytreeptrs.back());
    mytreeids.resize(mytreeptrs.back());

    IndexVector ptrs = mytreeptrs;

    for (const auto& [point, id, cell, ghost] : recvbuf)
    {
        Index slot = myslots.find(cell)->second;
        mytreeids[ptrs[slot]] = id;
        mytreepts[ptrs[slot]++] = point;
    }

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_POINT_ENVELOPE);
}
