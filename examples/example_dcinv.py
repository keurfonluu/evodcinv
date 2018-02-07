# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
try:
    from evoseispy import DispersionCurve
except ImportError:
    import sys
    sys.path.append("../")
    from evoseispy import DispersionCurve
    

if __name__ == "__main__":
    # Parameters
    disp_curve = np.loadtxt("data/dispersion_curve/func_mode.txt")
    ny = 100
    n_threads = 8
    n_runs = 1
    zmax = 15.
    
    # Inversion boundaries
    alpha_max = 1000.
    beta_min = [ 100., 100. ]
    beta_max = [ 500., 500. ]
    d_min = [ 1., 99999. ]
    d_max = [ 20., 99999. ]
    
    # Initialize dispersion curve
    phase_velocity = [ disp_curve[:,1] ]
    faxis = [ disp_curve[:,0] ]
    modes = [ 0 ]
    dc = DispersionCurve(phase_velocity, faxis, modes)

    # Evolutionary optimizer parameters
    evo_kws = dict(popsize = 20, max_iter = 100, constrain = True)
    opt_kws = dict(solver = "cmaes")
        
    # Multiple inversions
    dc.invert(beta_min, beta_max, d_min, d_max, alpha_max, ny = ny, n_runs = n_runs,
              n_threads = n_threads, verbose = 1, evo_kws = evo_kws, opt_kws = opt_kws)
        
    # Plot
    dc.plot(n_threads = n_threads)
    dc.plot_models(zmax = zmax)