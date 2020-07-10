#!/bin/bash

# MPI Parameters
MPIEXEC="/pp1/sw_tools/openmpi_4_0_2_for_pbspro/bin/mpiexec"
EXEC="python example_dcinv.py"

# Command
num_procs=20
num_threads=1
$MPIEXEC --hostfile $PBS_NODEFILE $EXEC
