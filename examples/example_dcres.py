# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import sys, glob
from argparse import ArgumentParser
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from evodcinv import LayeredModel, ThomsonHaskell, params2lay
except ImportError:
    sys.path.append("../")
    from evodcinv import LayeredModel, ThomsonHaskell, params2lay


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type = int, default = 8)
    args = parser.parse_args()
    
    # Parameters
    modes = [ 0, 1 ]
    wtypes = [ "rayleigh", "love" ]
    fmin, fmax, df = 0.1, 10., 0.1
    f = np.arange(fmin, fmax+df, df)
    ny = 200                            # Number of velocity discretization points
    perc = 90                           # Maximum RMS threshold as a percentage of best fitting models
    outdir = "output"                   # Output directory
    
    # Import inverted velocity models
    all_models, all_energy = [], []
    for filename in glob.glob("%s/run*.pickle" % outdir):
        m = pickle.load(open(filename, "rb"))
        all_models.append(np.hstack(m.models))
        all_energy.append(m.misfits.ravel())
    models = np.hstack(all_models).transpose()
    energy = np.hstack(all_energy)
    n_models = len(models)
    
    # Keep good fitting models only
    apost = np.exp(-0.5*energy**2)
    threshold = perc / 100. * apost.max()
    idx = np.where(apost > threshold)[0]
    models = models[idx]
    energy = energy[idx]
    
    # Sort models
    idx = np.argsort(energy)[::-1]
    models = models[idx]
    energy = energy[idx]
    pickle.dump(models, open("%s/models.pickle" % outdir, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    pickle.dump(energy, open("%s/energy.pickle" % outdir, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    # Recompute dispersion curves
    lay = np.array([ params2lay(m) for m in models ])
    
    if "rayleigh" in wtypes:
        rcurves = []
        for l in lay:
            th = ThomsonHaskell(l, wtype = "rayleigh")
            th.propagate(f, ny = ny, domain = "fc", n_threads = args.num_threads)
            rcurves.append(th.pick(modes = modes))
        pickle.dump(rcurves, open("%s/rcurves.pickle" % outdir, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    if "love" in wtypes:
        lcurves = []
        for l in lay:
            th = ThomsonHaskell(l, wtype = "love")
            th.propagate(f, ny = ny, domain = "fc", n_threads = args.num_threads)
            lcurves.append(th.pick(modes = modes))
        pickle.dump(lcurves, open("%s/lcurves.pickle" % outdir, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    # Print output statistics
    print("RMS min.: %.3f" % energy.min())
    print("RMS %.1f%%: %.3f" % (perc, np.sqrt(-2.*np.log(threshold))))
    print("Number of models kept: %d/%d (%.1f%%)" % (len(idx), n_models, len(idx)/n_models*100))