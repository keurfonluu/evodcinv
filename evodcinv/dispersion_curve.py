# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
    
__all__ = [ "DispersionCurve" ]

    
class DispersionCurve:
    
    _WTYPE = [ "rayleigh", "love" ]
    
    """
    Dispersion curve.
    
    Parameters
    ----------
    phase_velocity : list or ndarray
        Observed phase velocities (in m/s).
    faxis : list or ndarray
        Frequency axis (in Hz).
    mode : int
        Mode number (0 if fundamental).
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    """
    def __init__(self, phase_velocity, faxis, mode, wtype = "rayleigh"):
        if not isinstance(phase_velocity, (list, np.ndarray)) or np.asanyarray(phase_velocity).ndim != 1:
            raise ValueError("phase_velocity must be a list of 1-D ndarray")
        if not all([ np.min(c) > 0. for c in phase_velocity ]):
            raise ValueError("phase velocities must be positive")
        else:
            self._phase_velocity = phase_velocity
            self._npts = len(phase_velocity)
        if not isinstance(faxis, (list, np.ndarray)) or np.asanyarray(faxis).ndim != 1 \
            or len(faxis) != self._npts:
            raise ValueError("phase_velocity must be a list of 1-D ndarray of length %d" % self._npts)
        if not np.all([ np.min(f) >= 0. for f in faxis ]):
            raise ValueError("frequencies must be positive")
        else:
            self._faxis = faxis
        if not isinstance(mode, int) or mode < 0:
            raise ValueError("mode must be a positive integer")
        else:
            self._mode = mode
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype
            
    def save(self, filename = None, fmt = "%.8f"):
        """
        Export dispersion curve to ASCII file.
        
        Parameters
        ----------
        filename : str or None, default None
            Output file name.
        fmt : str, default "%.8f"
            ASCII format.
        """
        if filename is None:
            filename = "%s_mode%d.txt" % (self._wtype, self._mode)
        X = np.stack((self._faxis, self._phase_velocity), axis = 1)
        np.savetxt(filename, X, fmt)
            
    def plot(self, axes = None, figsize = (8, 8), plt_kws = {}):
        """
        Plot the dispersion curve.
        
        Parameters
        ----------
        axes : matplotlib axes or None, default None
            Axes used for plot.
        figsize : tuple, default (8, 8)
            Figure width and height if axes is None.
        plt_kws : dict
            Keyworded arguments passed to line plot.
            
        Returns
        -------
        lax : matplotlib line plot
            Line plot.
        """
        if axes is not None and not isinstance(axes, Axes):
            raise ValueError("axes must be Axes")
        if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple with 2 elements")
        if not isinstance(plt_kws, dict):
            raise ValueError("plt_kws must be a dictionary")
            
        if axes is None:
            fig = plt.figure(figsize = figsize, facecolor = "white")
            fig.patch.set_alpha(0.)
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = axes
        lax = ax1.plot(self._faxis, self._phase_velocity, **plt_kws)
        return lax
            
    @property
    def phase_velocity(self):
        """
        list or ndarray
        Observed phase velocities (in m/s).
        """
        return self._phase_velocity
    
    @phase_velocity.setter
    def phase_velocity(self, value):
        self._phase_velocity = value
        
    @property
    def faxis(self):
        """
        list or ndarray
        Frequency axis (in Hz).
        """
        return self._faxis
    
    @faxis.setter
    def faxis(self, value):
        self._faxis = value
        
    @property
    def mode(self):
        """
        int
        Mode number.
        """
        return self._mode
    
    @mode.setter
    def mode(self, value):
        self._mode = value
    
    @property
    def wtype(self):
        """
        str
        Surface wave type.
        """
        return self._wtype
    
    @wtype.setter
    def wtype(self, value):
        self._wtype = value
        
    @property
    def npts(self):
        """
        int
        Number of points that define the dispersion curve.
        """
        return self._npts