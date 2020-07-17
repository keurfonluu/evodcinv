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
    _DTYPE = [ "phase", "group" ]
    
    """
    Dispersion curve.
    
    Parameters
    ----------
    velocity : list or ndarray
        Observed velocities (in m/s).
    faxis : ndarray
        Frequency axis (in Hz).
    dtype: string. Accepted values: "phase" or "group".
        Choose between phase or group velocity dispersion curve.
    mode : int
        Mode number (0 if fundamental).
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    """
    def __init__(self, velocity, faxis, mode, wtype = "rayleigh", dtype="phase"):
        if not isinstance(velocity, (list, np.ndarray)) or np.asanyarray(velocity).ndim != 1:
            raise ValueError("velocity must be a list of 1-D ndarray")
        if not all([ np.min(c) > 0. for c in velocity ]):
            raise ValueError("phase velocities must be positive")
        else:
            self._velocity = velocity
            self._npts = len(velocity)
        if not isinstance(faxis, (list, np.ndarray)) or np.asanyarray(faxis).ndim != 1 \
            or len(faxis) != self._npts:
            raise ValueError("velocity must be a list of 1-D ndarray of length %d" % self._npts)
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

        self._period = np.sort(1. / self._faxis)
        self.dtype = dtype
            
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
        X = np.stack((self._faxis, self._velocity), axis = 1)
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
        lax = ax1.plot(self._faxis, self._velocity, **plt_kws)
        return lax
            
    @property
    def velocity(self):
        """
        list or ndarray
        Observed phase velocities (in m/s).
        """
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        self._velocity = value
        
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
    def period(self):
        """
        list or ndarray
        Period axis (in seconds).
        """
        return self._period
    
    @period.setter
    def faxis(self, value):
        self._period = value
        
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
    def dtype(self):
        """
        str
        Velocity axis type.
        Can be either "group" or "phase"
        """
        return self._dtype
    
    @dtype.setter
    def dtype(self, value):
        if value not in self._DTYPE:
            raise ValueError(
                    "Invalid value in dtype: {value}. Please input"  +
                    "one of the following: {DTYPE}."
            )
        self._dtype = value
        
    @property
    def npts(self):
        """
        int
        Number of points that define the dispersion curve.
        """
        return self._npts
