struct Hub
{
    static inline constexpr Distance distance = Distance();

    Index candidate, vertex, level;
    Real radius;

    IndexVector sites, ids, cells;
    RealVector dists;

    std::vector<Hub> children, leaves;

    Index size() const { return ids.size(); }

    void compute_child_hubs(const PointVector& points, Real cover, Index leaf_size, Real maxdist, Index& distcomps);
};

void Hub::compute_child_hubs(const PointVector& points, Real cover, Index leaf_size, Real maxdist, Index& distcomps)
{
    Real sep;
    Index m = size();

    Real target = std::pow(cover, (-level - 1.0)) * maxdist;

    do
    {
        Index new_candidate = candidate;
        Point new_candidate_point = points[new_candidate];
        sites.push_back(new_candidate);

        candidate = 0;
        sep = 0;

        for (Index j = 0; j < m; ++j)
        {
            Real d = distance(new_candidate_point, points[ids[j]]);

            if (d < dists[j])
            {
                dists[j] = d;
                cells[j] = new_candidate;
            }

            if (dists[j] > sep)
            {
                sep = dists[j];
                candidate = ids[j];
            }
        }

        distcomps += m;

    } while (sep > target);

    for (Index site : sites)
    {
        children.emplace_back();
        Hub& child = children.back();
        Index relcand = 0;

        for (Index j = 0; j < m; ++j)
        {
            if (cells[j] == site)
            {
                child.ids.push_back(ids[j]);
                child.cells.push_back(cells[j]);
                child.dists.push_back(dists[j]);

                if (child.dists.back() > child.dists[relcand])
                    relcand = child.dists.size()-1;
            }
        }

        child.sites.assign({site});
        child.candidate = child.ids[relcand];
        child.radius = child.dists[relcand];
        child.level = level+1;

        if (child.size() <= leaf_size || child.radius <= std::numeric_limits<Real>::epsilon())
        {
            leaves.push_back(child);
            children.pop_back();
        }
    }
}

void CoverTree::build(const PointVector& points, Real cover, Index leaf_size, Index& distcomps)
{
    struct BuildVertex
    {
        Index index;
        Real radius;
        IndexVector children, leaves;
        Index level;

        BuildVertex() {}
        BuildVertex(Index index, Real radius) : index(index), radius(radius) {}
    };

    n = points.size();
    std::deque<Hub> hubs;

    hubs.emplace_back();
    Hub& root_hub = hubs.back();

    root_hub.sites.assign({0});
    root_hub.ids.resize(n);
    root_hub.cells.resize(n, 0);
    root_hub.dists.resize(n);
    root_hub.radius = 0;
    root_hub.level = 0;

    for (Index i = 0; i < n; ++i)
    {
        root_hub.ids[i] = i;
        root_hub.dists[i] = distance(points[0], points[i]);

        if (root_hub.dists[i] > root_hub.radius)
        {
            root_hub.radius = root_hub.dists[i];
            root_hub.candidate = i;
        }
    }

    distcomps += n;

    Real maxdist = root_hub.radius;

    std::vector<BuildVertex> verts;

    verts.emplace_back(root_hub.sites.front(), root_hub.radius);
    root_hub.vertex = 0;

    Index num_children = 0;
    Index num_leaves = 0;

    while (!hubs.empty())
    {
        Hub hub = hubs.front(); hubs.pop_front();

        hub.compute_child_hubs(points, cover, leaf_size, maxdist, distcomps);

        for (Hub& child : hub.children)
        {
            Index vertex = verts.size();
            verts.emplace_back(child.sites.front(), child.radius);
            hubs.push_back(child);
            hubs.back().vertex = vertex;
            hubs.back().level = hub.level+1;
            verts.back().level = hub.level+1;
            verts[hub.vertex].children.push_back(vertex);
        }

        num_children += hub.children.size();

        for (Hub& leaf_hub : hub.leaves)
        {
            Index vertex = verts.size();
            verts.emplace_back(leaf_hub.sites.front(), leaf_hub.radius);
            verts.back().level = leaf_hub.level;

            verts[hub.vertex].children.push_back(vertex);
            num_children++;

            for (Index leaf : leaf_hub.ids)
            {
                verts[vertex].leaves.push_back(leaf);
            }

            num_leaves += leaf_hub.ids.size();
        }
    }

    vertices.reserve(verts.size());
    children.resize(num_children);
    leaves.resize(num_leaves);
    leaf_points.resize(num_leaves);

    Index child_ptr = 0;
    Index leaf_ptr = 0;

    for (const auto& [index, radius, mychildren, myleaves, level] : verts)
    {
        vertices.emplace_back();
        vertices.back().index = index;
        vertices.back().point = points[index];
        vertices.back().radius = radius;
        vertices.back().child_ptr = child_ptr;
        vertices.back().leaf_ptr = leaf_ptr;
        vertices.back().num_children = mychildren.size();
        vertices.back().num_leaves = myleaves.size();

        for (Index v : mychildren)
        {
            children[child_ptr++] = v;
        }

        for (Index l : myleaves)
        {
            leaf_points[leaf_ptr] = points[l];
            leaves[leaf_ptr++] = l;
        }
    }
}

Index CoverTree::range_query(IndexVector& neighbors, const Point& query, Real radius, Index& distcomps) const
{
    neighbors.clear();
    std::deque<Index> queue = {0};

    while (!queue.empty())
    {
        Index u = queue.front(); queue.pop_front();
        const auto& u_vtx = vertices[u];
        Index uid = u_vtx.index;

        for (Index i = u_vtx.leaf_ptr; i < u_vtx.leaf_ptr + u_vtx.num_leaves; ++i)
        {
            if (distance(query, leaf_points[i]) <= radius)
            {
                neighbors.push_back(leaves[i]);
            }
        }

        distcomps += u_vtx.num_leaves;

        for (Index i = u_vtx.child_ptr; i < u_vtx.child_ptr + u_vtx.num_children; ++i)
        {
            Index v = children[i];
            const auto& v_vtx = vertices[v];
            Index vid = v_vtx.index;
            Point vpt = v_vtx.point;

            if (distance(query, vpt) <= v_vtx.radius + radius)
            {
                queue.push_back(v);
            }
        }

        distcomps += u_vtx.num_children;
    }

    return neighbors.size();
}

template <class PointIter, class IndexIter>
void GhostTree::build(PointIter p1, PointIter p2, IndexIter i1, IndexIter i2, Index cellsize, Index site, Real cover, Index leaf_size, Index& distcomps)
{
    points.assign(p1, p2);
    ids.assign(i1, i2);
    this->site = site;
    tree.build(points, cover, leaf_size, distcomps);
    points.resize(cellsize);
}

Index GhostTree::graph_query(IndexVectorVector& graph, IndexVector& graphids, Real radius, Index& distcomps) const
{
    Index n_edges = 0;
    Index cellsize = points.size();

    for (Index i = 0; i < cellsize; ++i)
    {
        graph.emplace_back();
        graphids.push_back(ids[i]);
        n_edges += tree.range_query(graph.back(), points[i], radius, distcomps);
        std::for_each(graph.back().begin(), graph.back().end(), [&](Index& id) { id = ids[id]; });
    }

    return n_edges;
}

int GhostTree::get_packed_bufsize() const
{
    Index p = points.size();
    Index n = tree.n;
    Index l = tree.leaf_points.size();
    Index v = tree.vertices.size();
    Index c = tree.children.size();

    return sizeof(Index)*6 + sizeof(Index)*n + sizeof(Point)*p + sizeof(Point)*l + sizeof(Vertex)*v + sizeof(Index)*c + sizeof(Index)*l;
}

int GhostTree::pack_tree(char *buf, MPI_Comm comm) const
{
    MPI_Datatype MPI_VERTEX, MPI_POINT;

    MPI_Type_contiguous(DIM_SIZE, MPI_FLOAT, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    int blklens[4] = {1,1,1,4};
    MPI_Aint disps[4] = {offsetof(Vertex, index), offsetof(Vertex, point), offsetof(Vertex, radius), offsetof(Vertex, child_ptr)};
    MPI_Datatype types[4] = {MPI_INT64_T, MPI_POINT, MPI_FLOAT, MPI_INT64_T};
    MPI_Type_create_struct(4, blklens, disps, types, &MPI_VERTEX);
    MPI_Type_commit(&MPI_VERTEX);

    Index header[6];

    Index p = header[0] = points.size();
    Index n = header[1] = tree.n;
    Index l = header[2] = tree.leaf_points.size();
    Index v = header[3] = tree.vertices.size();
    Index c = header[4] = tree.children.size();
    header[5] = site;

    int bufsize = get_packed_bufsize();
    int position = 0;

    MPI_Pack(header, 6, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(ids.data(), n, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(points.data(), p, MPI_POINT, buf, bufsize, &position, comm);
    MPI_Pack(tree.leaf_points.data(), l, MPI_POINT, buf, bufsize, &position, comm);
    MPI_Pack(tree.vertices.data(), v, MPI_VERTEX, buf, bufsize, &position, comm);
    MPI_Pack(tree.children.data(), c, MPI_INT64_T, buf, bufsize, &position, comm);
    MPI_Pack(tree.leaves.data(), l, MPI_INT64_T, buf, bufsize, &position, comm);

    MPI_Type_free(&MPI_VERTEX);
    MPI_Type_free(&MPI_POINT);

    return bufsize;
}

void GhostTree::unpack_tree(const char *buf, int bufsize, MPI_Comm comm)
{
    MPI_Datatype MPI_VERTEX, MPI_POINT;

    MPI_Type_contiguous(DIM_SIZE, MPI_FLOAT, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);

    int blklens[4] = {1,1,1,4};
    MPI_Aint disps[4] = {offsetof(Vertex, index), offsetof(Vertex, point), offsetof(Vertex, radius), offsetof(Vertex, child_ptr)};
    MPI_Datatype types[4] = {MPI_INT64_T, MPI_POINT, MPI_FLOAT, MPI_INT64_T};
    MPI_Type_create_struct(4, blklens, disps, types, &MPI_VERTEX);
    MPI_Type_commit(&MPI_VERTEX);

    Index header[6];
    int position = 0;

    MPI_Unpack(buf, bufsize, &position, header, 6, MPI_INT64_T, comm);

    Index p = header[0];
    Index n = header[1];
    Index l = header[2];
    Index v = header[3];
    Index c = header[4];
    site = header[5];

    ids.resize(n);
    points.resize(p);
    tree.leaf_points.resize(l);
    tree.vertices.resize(v);
    tree.children.resize(c);
    tree.leaves.resize(l);
    tree.n = n;

    MPI_Unpack(buf, bufsize, &position, ids.data(), n, MPI_INT64_T, comm);
    MPI_Unpack(buf, bufsize, &position, points.data(), p, MPI_POINT, comm);
    MPI_Unpack(buf, bufsize, &position, tree.leaf_points.data(), l, MPI_POINT, comm);
    MPI_Unpack(buf, bufsize, &position, tree.vertices.data(), v, MPI_VERTEX, comm);
    MPI_Unpack(buf, bufsize, &position, tree.children.data(), c, MPI_INT64_T, comm);
    MPI_Unpack(buf, bufsize, &position, tree.leaves.data(), l, MPI_INT64_T, comm);

    MPI_Type_free(&MPI_VERTEX);
    MPI_Type_free(&MPI_POINT);
}

