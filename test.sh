#!/bin/bash

make clean && make -j12 D=40 LOG=0

./ptgen 10k p10k.fvecs
./ptgen 20k p20k.fvecs
./ptgen 40k p40k.fvecs
./ptgen 80k p80k.fvecs

export OMP_PLACES=cores

OMP_NUM_THREADS=1 ./rgraph -H p10k.fvecs 2
OMP_NUM_THREADS=2 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=3 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=4 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=5 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=6 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=7 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=8 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=9 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=10 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=11 ./rgraph    p10k.fvecs 2
OMP_NUM_THREADS=12 ./rgraph    p10k.fvecs 2

OMP_NUM_THREADS=1 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=2 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=3 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=4 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=5 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=6 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=7 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=8 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=9 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=10 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=11 ./rgraph    p20k.fvecs 1.95
OMP_NUM_THREADS=12 ./rgraph    p20k.fvecs 1.95

OMP_NUM_THREADS=1 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=2 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=3 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=4 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=5 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=6 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=7 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=8 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=9 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=10 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=11 ./rgraph    p40k.fvecs 1.85
OMP_NUM_THREADS=12 ./rgraph    p40k.fvecs 1.85
