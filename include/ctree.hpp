struct Hub
{
    static inline constexpr Distance distance = Distance();

    Index candidate, vertex, level;
    Real radius;

    IndexVector sites, ids, cells;
    RealVector dists;

    std::vector<Hub> children, leaves;

    Index size() const { return ids.size(); }

    void compute_child_hubs(const PointVector& points, Real cover, Index leaf_size, Real maxdist);
};

void Hub::compute_child_hubs(const PointVector& points, Real cover, Index leaf_size, Real maxdist)
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


template <class PointIter, class IndexIter>
void CoverTree::build(PointIter pfirst, PointIter plast, IndexIter ifirst, IndexIter ilast, Real cover, Index leaf_size)
{
    PointVector pts(pfirst, plast);
    ids.assign(ifirst, ilast);
    build(pts, cover, leaf_size);
}

void CoverTree::build(const PointVector& pts, Real cover, Index leaf_size)
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

    Index n = pts.size();
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
        root_hub.dists[i] = distance(pts[0], pts[i]);

        if (root_hub.dists[i] > root_hub.radius)
        {
            root_hub.radius = root_hub.dists[i];
            root_hub.candidate = i;
        }
    }

    Real maxdist = root_hub.radius;

    std::vector<BuildVertex> verts;

    verts.emplace_back(root_hub.sites.front(), root_hub.radius);
    root_hub.vertex = 0;

    Index num_children = 0;

    while (!hubs.empty())
    {
        Hub hub = hubs.front(); hubs.pop_front();

        hub.compute_child_hubs(pts, cover, leaf_size, maxdist);

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
        }
    }

    vertices.reserve(verts.size());
    children.resize(num_children);
    leaves.resize(n);
    points.resize(n);

    Index child_ptr = 0;
    Index leaf_ptr = 0;

    for (const auto& [index, radius, mychildren, myleaves, level] : verts)
    {
        vertices.emplace_back();
        vertices.back().index = index;
        vertices.back().point = pts[index];
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
            points[leaf_ptr] = pts[l];
            leaves[leaf_ptr++] = l;
        }
    }
}

void CoverTree::range_query(IndexVector& neighbors, const Point& query, Real radius) const
{
    std::deque<Index> queue = {0};

    while (!queue.empty())
    {
        Index u = queue.front(); queue.pop_front();
        const auto& u_vtx = vertices[u];
        Index uid = u_vtx.index;
        Point upt = u_vtx.point;

        for (Index i = u_vtx.leaf_ptr; i < u_vtx.leaf_ptr + u_vtx.num_leaves; ++i)
        {
            if (distance(query, points[i]) <= radius)
            {
                neighbors.push_back(leaves[i]);
            }
        }

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
    }

    if (!ids.empty()) for (Index& id : neighbors) id = ids[id];
}

void CoverTree::print_tree() const
{
    Index n = num_vertices();

    for (Index u = 0; u < n; ++u)
    {
        IndexVector cs(children.begin() + vertices[u].child_ptr, children.begin() + vertices[u].child_ptr + vertices[u].num_children);
        IndexVector ls(leaves.begin() + vertices[u].leaf_ptr, leaves.begin() + vertices[u].leaf_ptr + vertices[u].num_leaves);

        fmt::print("u={}\tid={}\tradius={:.3f}\tchildren={}\tleaves={}\n", u, vertices[u].index, vertices[u].radius, cs, ls);
    }
}
