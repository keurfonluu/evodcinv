# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import os, sys, time
from argparse import ArgumentParser
from copy import deepcopy
try:
    from mpi4py import MPI
    mpi_exist = True
except ImportError:
    mpi_exist = False
try:
    from evoseispy import DispersionCurve, progress_bar
except ImportError:
    sys.path.append("../")
    from evoseispy import DispersionCurve, progress_bar
    

if __name__ == "__main__":
    # Initialize MPI
    if mpi_exist:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_rank = 0
        
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type = int, default = 8)
    args = parser.parse_args()
        
    # Parameters
    ny = 500
    n_threads = args.num_threads
    max_run = 1
    outdir = "dc_output"
    
    # Inversion boundaries
    alpha_max = 8000.
#    beta_min = [ 100., 100., 1000., 1000., 3400. ]
#    beta_max = [ 2000., 2000., 3000., 3000., 3600. ]
#    d_min = [ 50., 50., 50., 500., 99999. ]
#    d_max = [ 500., 500., 500., 3000., 99999. ]
    beta_min = [ 100., 100., 100., 100., 1000., 1000., 1000., 1000., 3400. ]
    beta_max = [ 2000., 2000., 2000., 2000., 3000., 3000., 3000., 3000., 3600. ]
    d_min = [ 50., 50., 50., 50., 50., 50., 500., 500., 99999. ]
    d_max = [ 250., 250., 250., 250., 250., 250., 3000., 3000., 99999. ]
#    beta_min = [ 100., 100., 1000., 1500., 3000., 3950., 4800. ]
#    beta_max = [ 2000., 2000., 2000., 5000., 4000., 3950., 4800. ]
#    d_min = [ 50., 50., 50., 500., 1000., 25000., 99999. ]
#    d_max = [ 500., 500., 500., 3000., 3000., 28000., 99999. ]

    # Evolutionary optimizer parameters
    evo_kws = dict(popsize = 50, max_iter = 500, mpi = True)
    opt_kws = dict(solver = "cpso")
    
    # Initialize dispersion curve
    phase_velocity = [
        np.loadtxt("data/music/mode0_new.txt", usecols = [ 1 ]),
#        np.loadtxt("data/music/mode1_new.txt", usecols = [ 1 ]),
        np.loadtxt("data/music/mode1_HF_new.txt", usecols = [ 1 ]),
        np.loadtxt("data/music/mode2_new.txt", usecols = [ 1 ]),
#        np.loadtxt("data/music/mode3_new.txt", usecols = [ 1 ]),
        ]
    faxis = [
        np.loadtxt("data/music/mode0_new.txt", usecols = [ 0 ]),
#        np.loadtxt("data/music/mode1_new.txt", usecols = [ 0 ]),
        np.loadtxt("data/music/mode1_HF_new.txt", usecols = [ 0 ]),
        np.loadtxt("data/music/mode2_new.txt", usecols = [ 0 ]),
#        np.loadtxt("data/music/mode3_new.txt", usecols = [ 0 ]),
        ]
    modes = [ 0, 1, 2 ]
    dc = DispersionCurve(phase_velocity, faxis, modes)
    
    # Inversion
    if mpi_rank == 0:
        starttime = time.time()
        os.makedirs(outdir, exist_ok = True)
        print("\nInverting dispersion curves using %s (popsize = %d):" \
              % (opt_kws["solver"].upper(), evo_kws["popsize"]))
        progress_bar(-1, max_run)
        
    dispersion_curves = []
    for i in range(max_run):
        dc.invert(beta_min, beta_max, d_min, d_max, alpha_max, ny = ny,
                  n_threads = n_threads, evo_kws = evo_kws, opt_kws = opt_kws)
        if mpi_rank == 0:
            dc.save("%s/run%d.pickle" % (outdir, i+1))
            dispersion_curves.append(deepcopy(dc))
            progress_bar(i, max_run)
    
    if mpi_rank == 0:
        print("\nElapsed time: %.2f seconds\n" % (time.time() - starttime))
        misfits = [ d.misfit for d in dispersion_curves ]
        idx = np.argmin(misfits)
        print(dispersion_curves[idx])