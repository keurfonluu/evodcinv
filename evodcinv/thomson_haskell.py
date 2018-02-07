# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from ._dispcurve import dispcurve as dc

__all__ = [ "ThomsonHaskell" ]


class ThomsonHaskell:
    
    _WTYPE = [ "rayleigh", "love" ]
    
    """
    Thomson-Haskell propagator.
    
    This class computes the analytical dispersion curve modes in a stratified
    medium.
    
    Parameters
    ----------
    velocity_model : ndarray
        Velocity model. Each row defines the layer parameters in the following
        order: P-wave velocity (m/s), S-wave velocity (m/s), density (kg/m3)
        and thickness (m).
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    """
    def __init__(self, velocity_model, wtype = "rayleigh"):
        if not isinstance(velocity_model, np.ndarray) or velocity_model.ndim != 2:
            raise ValueError("velocity_model must be a 2-D ndarray")
        if velocity_model.shape[1] != 4:
            raise ValueError("velocity_model must have 4 columns")
        else:
            self._velocity_model = velocity_model
        ratio = velocity_model[:,0] / velocity_model[:,1]
        if np.any(ratio < 1.4) or np.any(ratio > 2.6):
            raise ValueError("non physical Vp/Vs ratio")
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype
    
    def propagate(self, f, y = None, ny = 100, domain = "fc", eps = 0.,
                  n_threads = 1):
        """
        Compute the analytical dispersion curve modes.
        
        Parameters
        ----------
        f : ndarray
            Frequency axis (in Hz).
        ny : int, default 100
            Number of samples on the Y axis.
        domain : {'fc', 'fk'}, default 'fc'
            Domain in which the dispersion curve is computed:
            - 'fc', phase velocity (m/s).
            - 'fk', wavenumber (rad/m).
        eps : float, default 0.1
            Percentage of expand for Y axis.
        n_threads : int, default 1
            Number of threads to pass to OpenMP.
            
        Returns
        -------
        panel : ndarray
            Dispersion curve panel.
        """
        # Check inputs
        if not isinstance(f, np.ndarray) or f.ndim != 1:
            raise ValueError("f must be a 1-D ndarray")
        if y is not None and (not isinstance(y, np.ndarray) or y.ndim != 1):
            raise ValueError("y must be a 1-D ndarray")
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(domain, str) or domain not in [ "fc", "fk" ]:
            raise ValueError("domain should either be 'fc' or 'fk'")
        
        # Import parameters
        alpha = self._velocity_model[:,0]
        beta = self._velocity_model[:,1]
        rho = self._velocity_model[:,2]
        d = self._velocity_model[:,3]
        
        # Thomson-Haskell method
        vr = self._rayleigh_velocity()
        vmin = max(0.1, np.floor(vr.min()))
        vmax = beta.max()
        dv = eps * (vmax - vmin)
        vmin = max(0.1, vmin - dv)
        if domain == "fc":
            y = np.linspace(vmin, vmax, ny)
            panel = dc.fcpanel(f, y, alpha, beta, rho, d,
                               wtype = self._wtype, n_threads = n_threads)
        elif domain == "fk":
            y = np.linspace(2.*np.pi*f[0]/vmax, 2.*np.pi*f[-1]/vmin, ny)
            panel = dc.fkpanel(f, y, alpha, beta, rho, d,
                               wtype = self._wtype, n_threads = n_threads)
        
        self._faxis = f
        self._yaxis = y
        self._panel = np.real(panel)
        self._domain = domain
        return panel
    
    def _poisson_ratio(self):
        alpha = self._velocity_model[:,0]
        beta = self._velocity_model[:,1]
        ksi = alpha**2 / beta**2
        return ( 1. - 0.5*ksi ) / ( 1. - ksi )
    
    def _rayleigh_velocity(self):
        beta = self._velocity_model[:,1]
        nu = self._poisson_ratio()
        return beta * ( 0.87 + 1.12 * nu ) / ( 1. + nu )
    
    def pick(self, modes = [ 0 ]):
        """
        Pick dispersion curve for different propagation mode.
        
        Parameters
        ----------
        modes : list of int, default [ 0 ]
            Modes number to pick (0 is fundamental mode).
            
        Returns
        -------
        dc : list of list
            Picked phase velocities in m/s. The ith element of the list
            corresponds to the dispersion curve of the ith mode.
        faxis : list of list
            Associated frequencies in Hz. The ith element of the list
            corresponds to the frequencies of the dispersion curve of the
            ith mode.
        """
        if not hasattr(self, "_panel"):
            raise ValueError("no propagation performed yet")
        if self._domain == "fk":
            raise ValueError("cannot perform picking in FK domain")
        if not isinstance(modes, list) or np.min(modes) < 0 \
            or not np.all([ isinstance(m, int) for m in modes ]):
            raise ValueError("modes must be a list of positive integers")
        
        modes = np.unique(modes)
        dc = [ [] for m in modes ]
        faxis = [ [] for m in modes ]
        for f in range(len(self._faxis)):
            tmp = self._panel[:,f] / np.max(np.abs(self._panel[:,f]))
            idx = np.where((tmp[:-1] * tmp[1:]) < 0.)[0]
            for i, m in enumerate(modes):
                if len(idx) >= m+1:
                    xr = [ tmp[idx[m]], tmp[idx[m]+1] ]
                    vr = [ self._yaxis[idx[m]], self._yaxis[idx[m]+1] ]
                    x = ( vr[0] * xr[1] - vr[1] * xr[0] ) / ( xr[1] - xr[0] )
                    dc[i].append(x)
                    faxis[i].append(self._faxis[f])
        return dc, faxis
    
    def save_picks(self, filename, modes = [ 0 ]):
        dc, faxis = self.pick(modes)
        fid = open(filename, "w")
        for m in modes:
            if m == 0:
                header = "# Fundamental mode\n"
            else:
                header = "# Mode %d\n" % m
            fid.write(header)
            
            d = dc[m]
            f = faxis[m]
            npts = len(d)
            for i in range(npts):
                fid.write(str(f[i]) + " " + str(d[i]) + "\n")
            fid.write("\n")
        fid.close()
    
    def plot(self, n_levels = 200, axes = None, figsize = (8, 8), cmap = None,
             cont_kws = {}):
        """
        Plot the dispersion curve panel.
        
        Parameters
        ----------
        n_levels: int, default 200
            Number of levels for contour.
        axes : matplotlib axes or None, default None
            Axes used for plot.
        figsize : tuple, default (8, 8)
            Figure width and height if axes is None.
        cmap : str, default "viridis"
            Colormap.
        cont_kws : dict
            Keyworded arguments passed to contour plot.
            
        Returns
        -------
        ax1 : matplotlib axes
            Axes used for plot.
        """
        if not hasattr(self, "_panel"):
            raise ValueError("no propagation performed yet")
        if cmap is None:
            cmap = self._set_cmap()
        if axes is None:
            fig = plt.figure(figsize = figsize, facecolor = "white")
            fig.patch.set_alpha(0.)
            ax1 = fig.add_subplot(1, 1, 1)
        else:
            ax1 = axes
        
        ax1.contourf(self._faxis, self._yaxis, np.log(np.abs(self._panel)),
                     n_levels, cmap = cmap, **cont_kws)
        ax1.set_xlabel("Frequency (Hz)", fontsize = 12)
        if self._domain == "fc":
            ax1.set_ylabel("Phase velocity (m/s)", fontsize = 12)
        elif self._domain == "fk":
            ax1.set_ylabel("Wavenumber (rad/m)", fontsize = 12)
        ax1.grid(True, linestyle = ":")
        return ax1
        
    def _set_cmap(self):
        import matplotlib.cm as cm
        if hasattr(cm, "viridis"):
            return "viridis"
        else:
            return "jet"
    
    @property
    def velocity_model(self):
        """
        ndarray
        Velocity model. Each row defines the layer parameters in the following
        order: P-wave velocity (m/s), S-wave velocity (m/s), density (kg/m3)
        and thickness (m).
        """
        return self._velocity_model
    
    @velocity_model.setter
    def velocity_model(self, value):
        self._velocity_model = value
        
    @property
    def wtype(self):
        """
        str
        Surface wave type ('rayleigh' or 'love').
        """
        return self._wtype
    
    @wtype.setter
    def wtype(self, value):
        self._wtype = value
        
    @property
    def faxis(self):
        """
        ndarray
        Frequency axis (in Hz).
        """
        return self._faxis
    
    @faxis.setter
    def faxis(self, value):
        self._faxis = value
        
    @property
    def yaxis(self):
        """
        ndarray
        Y axis:
        - Phase velocity (m/s) if domain = 'fc'.
        - Wavenumber (rad/m) if domain = 'fk'.
        """
        return self._yaxis
    
    @yaxis.setter
    def yaxis(self, value):
        self._yaxis = value
        
    @property
    def panel(self):
        """
        ndarray
        Dispersion curve panel.
        """
        return self._panel
    
    @panel.setter
    def panel(self, value):
        self._panel = value
        
    @property
    def domain(self):
        """
        Domain in which the dispersion curve is computed.
        """
        return self._domain
    
    @domain.setter
    def domain(self, value):
        self._domain = value