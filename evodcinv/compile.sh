#!/bin/bash

# PARAMETERS
F2PY="f2py"
FC="gfortran"
FFLAGS="-O3 -ffast-math -march=native -funroll-loops -fno-protect-parens -flto -fopenmp"
F90DIR="f90/"

# COMMANDS
$F2PY -c -m _dispcurve --fcompiler=$FC --f90flags="$FFLAGS" -lgomp "$F90DIR"dispcurve.f90
