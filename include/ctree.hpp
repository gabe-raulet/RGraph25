void CoverTree::build(Real covering_factor, Index leaf_size)
{
    struct Hub
    {
        IndexVector sub_sites;
        IndexVector leaves;
        IndexVector myids;
        IndexVector mycells;
        RealVector mydists;
        Real radius;
        Index candidate;
        Index parent;

        std::vector<Hub> children;

        Index size() const { return myids.size(); }

        void compute_child_hubs(const PointVector& points, Real covering_factor, Index leaf_size)
        {
            Real sep;
            Index m = size();

            do
            {
                Index new_candidate = candidate;
                Point new_candidate_point = points[new_candidate];
                sub_sites.push_back(new_candidate);

                candidate = 0;
                sep = 0;

                for (Index j = 0; j < m; ++j)
                {
                    Real d = distance(new_candidate_point, points[myids[j]]);

                    if (d < mydists[j])
                    {
                        mydists[j] = d;
                        mycells[j] = new_candidate;
                    }

                    if (mydists[j] > sep)
                    {
                        sep = mydists[j];
                        candidate = myids[j];
                    }
                }

            } while (sep > radius / covering_factor);

            for (Index sub_site : sub_sites)
            {
                children.emplace_back();
                Hub& new_hub = children.back();
                Index relcand = 0;

                for (Index j = 0; j < size(); ++j)
                {
                    if (mycells[j] == sub_site)
                    {
                        new_hub.myids.push_back(myids[j]);
                        new_hub.mycells.push_back(mycells[j]);
                        new_hub.mydists.push_back(mydists[j]);

                        if (new_hub.mydists.back() > new_hub.mydists[relcand])
                            relcand = new_hub.mydists.size()-1;
                    }
                }

                new_hub.sub_sites.assign({sub_site});
                new_hub.candidate = new_hub.myids[relcand];
                new_hub.radius = new_hub.mydists[relcand];

                if (new_hub.size() <= leaf_size || new_hub.radius <= std::numeric_limits<Real>::epsilon())
                {
                    std::copy(new_hub.myids.begin(), new_hub.myids.end(), std::back_inserter(leaves));
                    children.pop_back();
                }
            }
        }
    };

    struct BuildVertex
    {
        Index id;
        Real radius;
        IndexVector children, leaves;

        BuildVertex() {}
        BuildVertex(Index id, Real radius) : id(id), radius(radius) {}
    };

    Index n = num_points();
    Index leaf_count = 0;

    std::deque<Hub> hubs;

    hubs.emplace_back();
    Hub& root_hub = hubs.back();

    root_hub.sub_sites.assign({0});
    root_hub.myids.resize(n);
    root_hub.mycells.resize(n, 0);
    root_hub.mydists.resize(n);
    root_hub.radius = 0;
    root_hub.parent = -1;

    for (Index i = 0; i < n; ++i)
    {
        root_hub.myids[i] = i;
        root_hub.mydists[i] = distance(points[0], points[i]);

        if (root_hub.mydists[i] > root_hub.radius)
        {
            root_hub.radius = root_hub.mydists[i];
            root_hub.candidate = i;
        }
    }

    std::vector<BuildVertex> build_vertices;

    Index num_children = 0;
    Index num_leaves = 0;

    while (!hubs.empty())
    {
        Hub hub = hubs.front(); hubs.pop_front();

        hub.compute_child_hubs(points, covering_factor, leaf_size);

        Index vertex = build_vertices.size();
        build_vertices.emplace_back(hub.sub_sites.front(), hub.radius);

        if (hub.parent >= 0)
        {
            build_vertices[hub.parent].children.push_back(vertex);
            num_children++;
        }

        for (Hub& new_hub : hub.children)
        {
            new_hub.parent = vertex;
            hubs.push_back(new_hub);
        }

        for (Index leaf : hub.leaves)
        {
            build_vertices[vertex].leaves.push_back(leaf);
        }

        num_leaves += hub.leaves.size();
    }

    vertices.reserve(build_vertices.size());
    children.resize(num_children);
    leaves.resize(num_leaves);

    Index cptr = 0;
    Index lptr = 0;

    for (const auto& [id, radius, mychildren, myleaves] : build_vertices)
    {
        vertices.emplace_back();
        vertices.back().id = id;
        vertices.back().radius = radius;
        vertices.back().cptr = cptr;
        vertices.back().lptr = lptr;
        vertices.back().csize = mychildren.size();
        vertices.back().lsize = myleaves.size();

        for (Index v : mychildren)
            children[cptr++] = v;

        for (Index l : myleaves)
            leaves[lptr++] = l;
    }
}

void CoverTree::range_query(IndexVector& neighbors, const Point& query, Real radius) const
{
    IndexVector stack = {0};

    while (!stack.empty())
    {
        Index u = stack.back(); stack.pop_back();
        const auto& u_vtx = vertices[u];
        Index uid = u_vtx.id;
        Point upt = points[uid];

        for (Index i = u_vtx.lptr; i < u_vtx.lptr + u_vtx.lsize; ++i)
            if (distance(query, points[leaves[i]]) <= radius)
                neighbors.push_back(globids[leaves[i]]);

        for (Index i = u_vtx.cptr; i < u_vtx.cptr + u_vtx.csize; ++i)
        {
            Index v = children[i];
            const auto& v_vtx = vertices[v];
            Index vid = v_vtx.id;
            Point vpt = points[vid];

            if (distance(query, vpt) <= v_vtx.radius + radius)
                stack.push_back(v);
        }
    }
}

template <class PointIter>
void CoverTree::assign_points(PointIter first, PointIter last)
{
    points.assign(first, last);
    globids.resize(points.size());
    std::iota(globids.begin(), globids.end(), (Index)0);
    vertices.clear(), children.clear(), leaves.clear();
}

template <class PointIter, class IndexIter>
void CoverTree::assign_points(PointIter pfirst, PointIter plast, IndexIter ifirst, IndexIter ilast)
{
    points.assign(pfirst, plast);
    globids.assign(ifirst, ilast);
    vertices.clear(), children.clear(), leaves.clear();
}

int CoverTree::get_packed_bufsize() const
{
    Index p = num_points();
    Index v = num_vertices();
    Index c = children.size();
    Index l = leaves.size();

    return sizeof(Index)*5 + sizeof(Point)*p + sizeof(Index)*p + sizeof(Vertex)*v + sizeof(Index)*(c + l);
}

void CoverTree::unpack_tree(const char *buf, int bufsize, MPI_Comm comm)
{
    points.clear();
    vertices.clear();
    globids.clear();
    children.clear();
    leaves.clear();

    MPI_Datatype MPI_VERTEX, MPI_POINT;

    MPI_Type_contiguous(DIM_SIZE, MPI_FLOAT, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    int blklens[3] = {1,1,4};
    MPI_Aint disps[3] = {offsetof(Vertex, id), offsetof(Vertex, radius), offsetof(Vertex, cptr)};
    MPI_Datatype types[3] = {MPI_INT64_T, MPI_FLOAT, MPI_INT64_T};
    MPI_Type_create_struct(3, blklens, disps, types, &MPI_VERTEX);
    MPI_Type_commit(&MPI_VERTEX);

    Index header[5];

    int position = 0;

    MPI_Unpack(buf, bufsize, &position, header, 5, MPI_INT64_T, comm);

    Index p = header[0];
    Index v = header[1];
    Index c = header[2];
    Index l = header[3];
    site = header[4];

    points.resize(p);
    globids.resize(p);
    vertices.resize(v);
    children.resize(c);
    leaves.resize(l);

    MPI_Unpack(buf, bufsize, &position, points.data(), p, MPI_POINT, comm);
    MPI_Unpack(buf, bufsize, &position, vertices.data(), v, MPI_VERTEX, comm);
    MPI_Unpack(buf, bufsize, &position, globids.data(), p, MPI_INT64_T, comm);
    MPI_Unpack(buf, bufsize, &position, children.data(), c, MPI_INT64_T, comm);
    MPI_Unpack(buf, bufsize, &position, leaves.data(), l, MPI_INT64_T, comm);

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_VERTEX);
}

int CoverTree::pack_tree(char *buf, MPI_Comm comm) const
{
    MPI_Datatype MPI_VERTEX, MPI_POINT;

    MPI_Type_contiguous(DIM_SIZE, MPI_FLOAT, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    int blklens[3] = {1,1,4};
    MPI_Aint disps[3] = {offsetof(Vertex, id), offsetof(Vertex, radius), offsetof(Vertex, cptr)};
    MPI_Datatype types[3] = {MPI_INT64_T, MPI_FLOAT, MPI_INT64_T};
    MPI_Type_create_struct(3, blklens, disps, types, &MPI_VERTEX);
    MPI_Type_commit(&MPI_VERTEX);

    Index header[5];

    Index p = header[0] = num_points();
    Index v = header[1] = num_vertices();
    Index c = header[2] = children.size();
    Index l = header[3] = leaves.size();

    header[4] = site;

    int bufsize = get_packed_bufsize();

    int position = 0;

    MPI_Pack(header, 5, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(points.data(), p, MPI_POINT, buf, bufsize, &position, comm);
    MPI_Pack(vertices.data(), v, MPI_VERTEX, buf, bufsize, &position, comm);
    MPI_Pack(globids.data(), p, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(children.data(), c, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(leaves.data(), l, MPI_INT64_T, buf, bufsize, &position, comm);

    MPI_Type_free(&MPI_POINT);
    MPI_Type_free(&MPI_VERTEX);

    return bufsize;
}
