#!/bin/bash

EPSILON=$1

./ptgen 10k p10k.fvecs
./ptgen 20k p20k.fvecs
./ptgen 40k p40k.fvecs
./ptgen 80k p80k.fvecs

srun -N 1 --ntasks-per-node=4 ./rgraph_mpi -H p10k.fvecs 0.01 10

for DATASET in p10k.fvecs p20k.fvecs p40k.fvecs p80k.fvecs
do
    for NUM_RANKS in 1 2 4 8 16 32 64 128
    do
        srun -N 1 --ntasks-per-node=$NUM_RANKS --cpu_bind=cores ./rgraph_mpi $DATASET $EPSILON 128
    done
done
