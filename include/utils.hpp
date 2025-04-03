Real Distance::operator()(const Point& p, const Point& q) const
{
    Real sum = 0, delta;

    for (int i = 0; i < DIM_SIZE; ++i)
    {
        delta = p[i] - q[i];
        sum += delta * delta;
    }

    return std::sqrt(sum);
}

template <class Integer>
void get_balanced_counts(std::vector<Integer>& counts, size_t totsize)
{
    Integer blocks = counts.size();
    std::fill(counts.begin(), counts.end(), totsize/blocks);

    counts.back() = totsize - (blocks-1)*(totsize/blocks);

    Integer diff = counts.back() - counts.front();

    for (Integer i = 0; i < diff; ++i)
    {
        counts[blocks-1-i]++;
        counts[blocks-1]--;
    }
}

void read_fvecs(PointVector& points, const char *fname)
{
    Point p;
    std::ifstream is;
    PointRecord record;
    size_t filesize, n;
    int dim;

    is.open(fname, std::ios::binary | std::ios::in);

    is.seekg(0, is.end);
    filesize = is.tellg();
    is.seekg(0, is.beg);

    is.read((char*)&dim, sizeof(int)); assert((dim == DIM_SIZE));
    is.seekg(0, is.beg);

    assert((filesize % sizeof(PointRecord)) == 0);
    n = filesize / sizeof(PointRecord);
    points.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
        is.read(record.data(), sizeof(PointRecord));

        const char *dim_src = record.data();
        const char *pt_src = record.data() + sizeof(int);
        char *pt_dest = (char*)points[i].data();

        std::memcpy(&dim, dim_src, sizeof(int)); assert((dim == DIM_SIZE));
        std::memcpy(pt_dest, pt_src, sizeof(Point));
    }

    is.close();
}

void read_fvecs(PointVector& mypoints, Index& myoffset, Index& totsize, const char *fname, MPI_Comm comm)
{
    int myrank, nprocs;
    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &nprocs);

    PointRecord record;
    uint64_t b[2];
    int dim;

    uint64_t& filesize = b[0], &n = b[1];

    if (!myrank)
    {
        std::ifstream is;
        is.open(fname, std::ios::binary | std::ios::in);

        is.seekg(0, is.end);
        filesize = is.tellg();
        is.seekg(0, is.beg);

        is.read((char*)&dim, sizeof(int)); assert((dim == DIM_SIZE));
        is.close();

        assert((filesize % sizeof(PointRecord)) == 0);
        n = filesize / sizeof(PointRecord);
    }

    MPI_Bcast(b, 2, MPI_UINT64_T, 0, comm);
    dim = DIM_SIZE;

    IndexVector counts(nprocs, n/nprocs), displs(nprocs);
    counts.back() = n - (nprocs-1)*(n/nprocs);

    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), static_cast<Index>(0));

    Index mysize = counts[myrank];
    myoffset = displs[myrank];
    totsize = n;

    std::vector<char> mybuf(mysize * sizeof(PointRecord));

    MPI_File fh;
    MPI_File_open(comm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_Offset fileoffset = myoffset * sizeof(PointRecord);
    MPI_File_read_at_all(fh, fileoffset, mybuf.data(), static_cast<int>(mybuf.size()), MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

    mypoints.resize(mysize);
    char *ptr = mybuf.data();

    for (Index i = 0; i < mysize; ++i)
    {
        std::memcpy(record.data(), ptr, sizeof(PointRecord));

        const char *dim_src = record.data();
        const char *pt_src = record.data() + sizeof(int);
        char *pt_dest = (char*)mypoints[i].data();

        std::memcpy(&dim, dim_src, sizeof(int)); assert((dim == DIM_SIZE));
        std::memcpy(pt_dest, pt_src, sizeof(Point));
        ptr += sizeof(PointRecord);
    }
}

void write_fvecs(const PointVector& points, const char *fname)
{
    std::ofstream os;
    PointRecord record;
    int dim = DIM_SIZE;

    char *dim_dest = record.data();
    char *pt_dest = dim_dest + sizeof(int);

    std::memcpy(dim_dest, &dim, sizeof(int));

    os.open(fname, std::ios::binary | std::ios::out);

    for (const Point& p : points)
    {
        char *pt_src = (char*)p.data();
        std::memcpy(pt_dest, pt_src, sizeof(Point));
        os.write(record.data(), sizeof(PointRecord));
    }

    os.close();
}

bool check_correctness(const PointVector& points, IndexVectorVector& graph, Real epsilon)
{
    Distance distance;

    bool correct = true;
    Index n = points.size();

    assert((n == points.size()));

    #pragma omp parallel for shared(correct)
    for (Index i = 0; i < n; ++i)
    {
        if (!correct) continue;

        IndexVector neighbors;
        neighbors.reserve(graph[i].size());

        for (Index j = 0; j < n; ++j)
            if (distance(points[i], points[j]) <= epsilon)
                neighbors.push_back(j);

        if (graph[i].size() != neighbors.size())
        {
            #pragma omp atomic write
            correct = false;
        }
        else
        {
            std::sort(graph[i].begin(), graph[i].end());
            std::sort(neighbors.begin(), neighbors.end());

            if (graph[i] != neighbors)
            {
                #pragma omp atomic write
                correct = false;
            }
        }
    }

    return correct;
}
