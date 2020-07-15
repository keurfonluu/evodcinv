#!/bin/bash

# MPI Parameters
MPIEXEC="/pp1/sw_tools/openmpi_4_0_2_for_pbspro/bin/mpiexec" # --tag-output" ##Option specifies which node is running
EXEC="python example_dcinv.py"

# Command
$MPIEXEC --hostfile $PBS_NODEFILE $EXEC
