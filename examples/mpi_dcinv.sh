#!/bin/bash

# MPI Parameters
MPIEXEC="mpiexec"
EXEC="python example_dcinv.py"

# Command
read -p "Number of MPI process: " num_procs
read -p "Number of threads per process: " num_threads
$MPIEXEC -np $num_procs --bind-to core:overload-allowed --map-by ppr:$num_procs:node:pe=$num_threads $EXEC -n $num_threads
