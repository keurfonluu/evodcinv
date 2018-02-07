# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


def on_pick(event):
    x, y = event.mouseevent.xdata, event.mouseevent.ydata
    f_picks.append(x)
    k_picks.append(y)
    print(x, y)


if __name__ == "__main__":
    # Parameters
    faxis = np.fromfile("data/faxis.bin", dtype = "float64")
    kaxis = np.fromfile("data/kaxis.bin", dtype = "float64")
    fk = np.fromfile("data/fk.bin", dtype = "float64")
    
    # Picks
    f_picks = []
    k_picks = []
    fmode_0 = np.loadtxt("data/mode_0_anc.txt", usecols = [ 0 ])
    kmode_0 = np.loadtxt("data/mode_0_anc.txt", usecols = [ 1 ])
    fmode_1 = np.loadtxt("data/mode_1_anc.txt", usecols = [ 0 ])
    kmode_1 = np.loadtxt("data/mode_1_anc.txt", usecols = [ 1 ])
    
    # Reshape FK
    nf = len(faxis)
    nk = len(kaxis)
    fk = fk.reshape((nf, nk), order = "F")
    
    # Plot
    fig = plt.figure(figsize = (5, 5), facecolor = "white")
    fig.patch.set_alpha(0.)
    fig.canvas.mpl_connect("pick_event", on_pick)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_picker(True)
    
    ax1.contourf(faxis, kaxis, fk.T, 200)
    ax1.plot(fmode_0, kmode_0, linewidth = 2, color = "red")
    ax1.plot(fmode_1, kmode_1, linewidth = 2, color = "red")
    ax1.set_ylim(-0.03, 0.03)
    ax1.set_xlabel("Frequency (Hz)", fontsize = 12)
    ax1.set_ylabel("Wavenumber (rad/m)", fontsize = 12)
    ax1.grid(True, linestyle = ":")
    fig.tight_layout()