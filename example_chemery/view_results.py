# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from evoseispy import DispersionCurve
except ImportError:
    sys.path.append("../")
    from evoseispy import DispersionCurve


def mad(x):
    return np.median(np.abs(x - np.median(x)))


if __name__ == "__main__":
    # Parameters
    outdir = "dc_output/layer9"
    max_run = 20
    zmax, nz = 10000, 1000
    prctile = [ 2.5, 97.5 ]
    blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
    green = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
    
    # Import VP log
    dp, vp = np.loadtxt("data/vp_log.txt", unpack = True)
    
    # Import files
    dispersion_curves = []
    all_models = []
    all_energy = []
    for i in range(max_run):
        filename = "%s/run%d.pickle" % (outdir, i+1)
        dc = pickle.load(open(filename, "rb"))
        dispersion_curves.append(dc)
        all_models.append(np.hstack(dc.models))
        all_energy.append(dc.misfits.ravel(order = "F"))
        
    models = np.hstack(all_models).transpose()
    energy = np.hstack(all_energy)
    
    # Sort according to energies
    idx = np.argsort(energy)[::-1]
    energy = energy[idx]
    models = models[idx]
    
    # Remove very bad models
    idx = energy < 100000
#    idx = np.logical_and(idx, [ np.all(m[:10] > 0) for m in models ])
    energy = energy[idx]
    models = models[idx]
    
    # Make colormap
    cmap = "jet_r"
    energy_normed = ( energy - energy.min() ) \
                     / ( energy.max() - energy.min() )
    colors = eval("cm.%s(energy_normed)" % cmap)
    
    # Convert to continuous velocity models
    lay = [ dc.params2vel(l) for l in models ]
    vel = np.array([ dc._make_vel(l[:,1], l[:,3], zmax, nz) for l in lay ])
        
    idx = np.argmin(energy)
    pbest = vel[idx]
    pmean = np.average(vel, 0, weights = 1./energy)
    lower, upper = np.transpose([ np.percentile(v, prctile) for v in vel.transpose() ])
        
    # Plot
    sns.set(font_scale = 2, rc = {"legend.fontsize": 16})
    sns.set_style("ticks")
    
    fig1 = plt.figure(figsize = (5, 6), facecolor = "white")
    ax1 = fig1.add_subplot(1, 1, 1)
    fig1.patch.set_alpha(0.)
    
    az = np.linspace(0., zmax, nz)
    ax1.fill_betweenx(az, lower, upper, color = blue, alpha = 0.15)
    ax1.plot(gaussian_filter(vp/2, 20), dp, linewidth = 0.5, color = "black", label = "Sonic")
    ax1.plot(pmean, az, color = blue, label = "Mean")
    ax1.plot(pbest, az, color = "red", label = "Best")
    
    ax1.get_xaxis().set_tick_params(direction = "in")
    ax1.get_yaxis().set_tick_params(direction = "in")
    ax1.set_ylim(0., 5000.)
    ax1.set_xlabel("$V_S$ (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle = ":")
    ax1.legend(loc = 3)
    fig1.tight_layout()
    
    # Dispersion curves in best model
    idx = np.argmin([ dc.misfit for dc in dispersion_curves ])
    dc = dispersion_curves[idx]
    out = dc.pick(nf = 100)
    
    fig2 = plt.figure(figsize = (5, 5), facecolor = "white")
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.patch.set_alpha(0.)
    
    for d, f in zip(*out):
        ax2.plot(f, d, linewidth = 2, color = "black")
        
    for f, c in zip(dc.faxis, dc.phase_velocity):
        ax2.scatter(f, c, s = 100, marker = "+", color = "red")
        
    ax2.set_xlabel("Frequency (Hz)", fontsize = 12)
    ax2.set_ylabel("Phase velocity (m/s)", fontsize = 12)
    ax2.grid(True, linestyle = ":")
    fig2.tight_layout()