#!/bin/bash

EPSILON=$1

make clean
make -j4 D=78 LOG=0

./ptgen 1k dummy

srun -N 1 --ntasks-per-node=4 ./rgraph_mpi -H dummy 0.01 10

for DATASET in datasets/twitter.fvecs
do
    #for NUM_RANKS in 1 2 4 8 16 32 64 128
    for NUM_RANKS in 8 16 32 64 128
    do
        for NUM_SITES in 512 1024 2048
        do
            srun -N 4 --ntasks-per-node=$NUM_RANKS --cpu_bind=cores ./rgraph_mpi -c 1.08 -R $DATASET $EPSILON $NUM_SITES
        done
    done
done

rm dummy
