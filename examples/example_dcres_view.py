# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from evodcinv import params2vel
except ImportError:
    sys.path.append("../")
    from evodcinv import params2vel


if __name__ == "__main__":
    # Parameters
    wtypes = [ "rayleigh", "love" ]
    fmin, fmax = 0.1, 10.
    skip = 50
    zmax = 1500
    outdir = "output"                   # Output directory
    cmap = "viridis_r"                  # Colormap
    
    # Unpickle required files
    models = pickle.load(open("%s/models.pickle" % outdir, "rb"))[::skip]
    energy = pickle.load(open("%s/energy.pickle" % outdir, "rb"))[::skip]
    if "rayleigh" in wtypes:
        rcurves = pickle.load(open("%s/rcurves.pickle" % outdir, "rb"))[::skip]
    if "love" in wtypes:
        lcurves = pickle.load(open("%s/lcurves.pickle" % outdir, "rb"))[::skip]
    
    # Convert acceptable models to continuous velocity models
    vel, az = np.transpose([ params2vel(m, zmax = zmax) for m in models ], axes = [ 1, 0, 2 ])
    
    # Import true models
    r0 = np.loadtxt("data/rayleigh_mode0.txt", unpack = True)
    r1 = np.loadtxt("data/rayleigh_mode1.txt", unpack = True)
    l0 = np.loadtxt("data/love_mode0.txt", unpack = True)
    l1 = np.loadtxt("data/love_mode1.txt", unpack = True)
    
    # Initialize figures
    fig1 = plt.figure(figsize = (5, 5), facecolor = "white")
    fig2 = plt.figure(figsize = (5*len(wtypes), 5), facecolor = "white")
    fig1.patch.set_alpha(0.)
    fig2.patch.set_alpha(0.)
    ax1 = fig1.add_subplot(1, 1, 1)
    ax2 = [ fig2.add_subplot(1, len(wtypes), i+1) for i, w in enumerate(wtypes) ]
    
    # Make colormap
    norm = Normalize(energy.min(), energy.max())
    smap = ScalarMappable(norm, cmap)
    smap.set_array([])
    
    # Plot velocity models
    for v, a, e in zip(vel, az, energy):
        ax1.plot(v, a, color = smap.to_rgba(e))
    
    ax1.set_xlabel("Velocity (m/s)", fontsize = 12)
    ax1.set_ylabel("Depth (m)", fontsize = 12)
    ax1.set_ylim(a[0], a[-1])
    ax1.invert_yaxis()
    ax1.grid(True, linestyle = ":")
    
    cb1 = fig1.colorbar(smap)
    cb1.set_label("RMS", fontsize = 12)
    
    # PLot dispersion curves
    if "rayleigh" in wtypes:
        ax = ax2[max(0, len(wtypes)-2)]
        for dcurves, e in zip(rcurves, energy):
            for dcurve in dcurves:
                dcurve.plot(axes = ax, plt_kws = dict(color = smap.to_rgba(e)))
        ax.scatter(r0[0], r0[1], s = 10, marker = "+", facecolor = "black", zorder = 10)
        ax.scatter(r1[0], r1[1], s = 10, marker = "+", facecolor = "black", zorder = 10)
        ax.set_title("Rayleigh-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize = 12)
        ax.set_ylabel("Phase velocity (m/s)", fontsize = 12)
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle = ":")
                
    if "love" in wtypes:
        ax = ax2[max(0, len(wtypes)-1)]
        for dcurves, e in zip(lcurves, energy):
            for dcurve in dcurves:
                dcurve.plot(axes = ax, plt_kws = dict(color = smap.to_rgba(e)))
        ax.scatter(l0[0], l0[1], s = 10, marker = "+", facecolor = "black", zorder = 10)
        ax.scatter(l1[0], l1[1], s = 10, marker = "+", facecolor = "black", zorder = 10)
        ax.set_title("Love-wave")
        ax.set_xlabel("Frequency (Hz)", fontsize = 12)
        ax.set_ylabel("Phase velocity (m/s)", fontsize = 12)
        ax.set_xlim(fmin, fmax)
        ax.grid(True, linestyle = ":")
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.show()
    fig2.show()