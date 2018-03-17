#!/bin/bash

# PARAMETERS
F2PY="f2py"
FC="gfortran"
FFLAGS="-O3 -ffast-math -funroll-loops -fno-protect-parens -fopenmp"
F90DIR="f90/"

# COMMANDS
$F2PY -c -m _dispcurve --fcompiler=$FC --f90flags="$FFLAGS" -lgomp "$F90DIR"dispcurve.f90
$F2PY -c -m _lay2vel --fcompiler=$FC --f90flags="$FFLAGS" "$F90DIR"lay2vel.f90
