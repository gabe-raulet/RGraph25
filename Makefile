DEBUG?=0
LOG?=1
D?=32
STATS?=1
FLAGS=-std=c++20 -fopenmp
INCS=-I./ -I./include

ifeq ($(shell uname -s),Linux)
COMPILER=CC
MPI_COMPILER=CC
else
COMPILER=clang++
MPI_COMPILER=mpic++
endif

ifeq ($(DEBUG),1)
FLAGS+=-O0 -g -fsanitize=address -fno-omit-frame-pointer -DDEBUG
else
FLAGS+=-O2
endif

ifeq ($(LOG),1)
FLAGS+=-DLOG
endif

ifeq ($(STATS),1)
FLAGS+=-DSTATS
endif

all: rgraph rgraph_mpi perftest ptgen

rgraph: rgraph.cpp include
	$(MPI_COMPILER) -o $@ -DDIM_SIZE=$(D) $(FLAGS) $(INCS) $<

rgraph_mpi: rgraph_mpi.cpp include
	$(MPI_COMPILER) -o $@ -DDIM_SIZE=$(D) $(FLAGS) $(INCS) $<

perftest: perftest.cpp include
	$(MPI_COMPILER) -o $@ -DDIM_SIZE=$(D) $(FLAGS) $(INCS) $<

ptgen: ptgen.cpp include
	$(MPI_COMPILER) -o $@ -DDIM_SIZE=$(D) $(FLAGS) $(INCS) $<

clean:
	rm -rf rgraph rgraph_mpi ptgen perftest *.out *.dSYM
