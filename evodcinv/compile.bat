@echo OFF

:: PARAMETERS
set F2PY=f2py
set FC=gfortran
set FFLAGS="-O3 -ffast-math -march=native -funroll-loops -fno-protect-parens -flto -fopenmp"
set F90DIR=f90/

:: COMMANDS
call %F2PY% -c -m _dispcurve --fcompiler=%FC% --f90flags=%FFLAGS% -lgomp %F90DIR%dispcurve.f90
