# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fteikpy import lay2vel
from stochopy import Evolutionary
from .thomson_haskell import ThomsonHaskell
try:
    import cPickle as pickle
except ImportError:
    import pickle
    
__all__ = [ "StratifiedModel" ]

    
class StratifiedModel:
    """
    Stratified velocity model
    
    This class inverts for a stratified medium given different modes of an
    observed dispersion curve.
    
    Parameters
    ----------
    phase_velocity : list of ndarray
        Observed phase velocities (in m/s). The ith element corresponds to the
        ith mode in modes.
    faxis : list of ndarray
        Frequency axis (in Hz). The ith element corresponds to the ith mode
        in modes.
    modes : list of int
        Mode numbers associated to elements in phase_velocity in faxis.
    wtype : {'rayleigh', 'love'}, default 'rayleigh'
        Surface wave type.
    """
    
    def __init__(self, phase_velocity, faxis, modes, wtype = "rayleigh"):
        if not isinstance(phase_velocity, list) \
            or not np.all([ isinstance(c, np.ndarray) for c in phase_velocity ]):
            raise ValueError("phase_velocity must be a list of ndarray")
        if not all([ np.min(c) > 0. for c in phase_velocity ]):
            raise ValueError("phase velocities must be positive")
        else:
            self._n_modes = len(phase_velocity)
            self._phase_velocity = phase_velocity
        if not isinstance(faxis, list) \
            or not np.all([ isinstance(f, np.ndarray) for f in faxis ]):
            raise ValueError("faxis must be a list of ndarray")
        if len(faxis) != self._n_modes:
            raise ValueError("inconsistent number of modes in faxis, got %d instead of %d" \
                             % (len(faxis), self._n_modes))
        if not np.all([ len(f) == len(c) for f, c in zip(phase_velocity, faxis) ]):
            raise ValueError("inconsistent lengths for elements in phase_velocity and faxis")
        if not np.all([ np.min(f) > 0. for f in faxis ]):
            raise ValueError("frequencies must be positive")
        else:
            self._faxis = faxis
        if not isinstance(modes, list) \
            or not np.all([ isinstance(m, int) for m in modes ]):
            raise ValueError("modes must be a list of int")
        if len(modes) != self._n_modes:
            raise ValueError("inconsistent number of modes in modes, got %d instead of %d" \
                             % (len(modes), self._n_modes))
        if not np.all([ m >= 0 for m in modes ]):
            raise ValueError("modes must be positive")
        else:
            self._modes = modes
        if wtype not in self._WTYPE:
            raise ValueError("wtype must be in %s, got '%s'" % (self._WTYPE, wtype))
        else:
            self._wtype = wtype
            
    def __str__(self):
        model = "%s:\n%s" % ("model".rjust(13), self._print_attr("model"))
        misfit = "%s: %s" % ("misfit".rjust(13), self._print_attr("misfit"))
        n_iter = "%s: %s" % ("n_iter".rjust(13), self._print_attr("n_iter"))
        n_eval = "%s: %s" % ("n_eval".rjust(13), self._print_attr("n_eval"))
        return "\n".join((model, misfit, n_iter, n_eval)) + "\n"
        
    def _print_attr(self, attr):
        if attr not in [ "model", "misfit", "n_iter", "n_eval", "sigma" ]:
            raise ValueError("attr should be 'model', 'misfit', 'n_iter', 'n_eval' or 'sigma'")
        else:
            if attr == "model":
                n_lay = len(self._model) // 3
                model = self._model.reshape((n_lay, 3), order = "F")
                param = "\t\tVP (m/s)\tVS (m/s)\tThickness (m)\n"
                for i in range(n_lay):
                    param += "\t\t%.2f\t\t%.2f\t\t%.2f\n" % (model[i,0]*model[i,2], model[i,0], model[i,1])
                return param[:-2]
            elif attr == "misfit":
                return "%.2f" % self._misfit
            elif attr == "n_iter":
                return "%s" % self._n_iter
            elif attr == "n_eval":
                return "%s" % self._n_eval
    
    def invert(self, beta, thickness, alpha_max = None, ny = 100,
               dtype = "float32", n_threads = 1,
               evo_kws = dict(popsize = 10, max_iter = 100, constrain = True),
               opt_kws = dict(solver = "cpso")):
        """
        Invert the different modes of the dispersion curve for a stratified
        velocity model. Layers' P-wave velocities are determined by the Vp/Vs
        ratio ranging in [ 1.4, 2.6 ]. High uncertainties should be expected
        for P-wave velocities as surface waves are not much sensitive to Vp.
        Layers' densities are determined using the Nafe-Drake's equation as
        they only affect the amplitudes of the dispersion, not the location of
        the zero-crossing.
        
        Parameters
        ----------
        beta : ndarray (beta_min, beta_max)
            S-wave velocity boundaries in m/s.
        thickness : ndarray (d_min, d_max)
            Layer thickness boundaries in m.
        alpha_max : float, default None
            P-wave velocity maximum value in m/s. Models with higher P-wave
            velocities are clipped to alpha_max.
        ny : int, default 100
            Number of samples on the Y axis.
        dtype : {'float32', 'float64'}, default 'float32'
            Models data type.
        n_threads : int, default 1
            Number of threads to pass to OpenMP for forward modelling.
        evo_kws : dict
            Keywords to pass to evolutionary algorithm initialization.
        opt_kws : dict
            Keywords to pass to evolutionary algorithm optimizer.
        """
        if not isinstance(beta, np.ndarray) or beta.ndim != 2:
            raise ValueError("beta must be a 2-D ndarray")
        else:
            self._n_layers = beta.shape[0]
        if np.any(beta[:,1] < beta[:,0]):
            raise ValueError("elements in beta_max must be greater than beta_min")
        if not isinstance(thickness, np.ndarray) or thickness.ndim != 2:
            raise ValueError("thickness must be a 2-D ndarray")
        if thickness.shape[0] != self._n_layers:
            raise ValueError("inconsistent number of layers in thickness, got %d instead of %d" \
                             % (thickness.shape[0], self._n_layers))
        if np.any(thickness[:,1] < thickness[:,0]):
            raise ValueError("elements in d_max must be greater than d_min")
        if alpha_max is not None and not isinstance(alpha_max, (float, int)) or alpha_max < 1.4*np.max(beta[:,1]):
            raise ValueError("alpha_max must be a float greater than %.2f" % 1.4*np.max(beta[:,1]))
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError("n_threads must be a positive integer")
        if not isinstance(opt_kws, dict):
            raise ValueError("opt_kws must be a dictionary")
        if not isinstance(evo_kws, dict):
            raise ValueError("evo_kws must be a dictionary")
        if "constrain" not in evo_kws:
            evo_kws.update(constrain = True)
        else:
            evo_kws["constrain"] = True
        if "eps2" not in evo_kws:
            evo_kws.update(eps2 = -1e30)
        else:
            evo_kws["eps2"] = -1e30
        if "snap" not in evo_kws:
            evo_kws.update(snap = True)
        else:
            evo_kws["snap"] = True
        
        args = (alpha_max, ny, n_threads)
        lower = np.hstack((beta[:,0], thickness[:,0], np.full(self._n_layers, 1.41)))
        upper = np.hstack((beta[:,1], thickness[:,1], np.full(self._n_layers, 2.59)))
        ea = Evolutionary(self._costfunc, lower, upper, args = args, **evo_kws)
        xopt, gfit = ea.optimize(**opt_kws)
        self._model = np.array(xopt, dtype = dtype)
        self._misfit = gfit
        self._models = np.array(ea.models, dtype = dtype)
        self._misfits = np.array(ea.energy, dtype = dtype)
        self._n_iter = ea.n_iter
        self._n_eval = ea.n_eval
        return self
    
    def _costfunc(self, x, *args):
        alpha_max, ny, n_threads = args
        vel = self.params2vel(x, alpha_max)
        th = ThomsonHaskell(vel, self._wtype)
        misfit = 0.
        for i, m in enumerate(self._modes):
            th.propagate(self._faxis[i], ny = ny, domain = "fc", n_threads = n_threads)
            if np.any([ np.isnan(sec) for sec in th._panel.ravel() ]):
                return np.Inf
            else:
                dc_calc, f_calc = th.pick([ m ])
                if len(dc_calc[0]) > 0:
                    dc_obs = np.interp(f_calc[0], self._faxis[i], self._phase_velocity[i])
                    misfit += np.sqrt(np.mean(np.square(dc_obs - dc_calc[0])))
                else:
                    misfit += np.Inf
                    break
        return misfit / len(self._modes)
    
    def _betanu2alpha(self, beta, nu):
        return beta * np.sqrt( np.abs( ( 1.-nu ) / ( 0.5 - nu ) ) ) 
    
    def _nafe_drake(self, alpha):
        coeff = np.array([ 1.6612, -0.4712, 0.0671, -0.0043, 0.000106 ])
        alpha_pow = np.array([ alpha*1e-3, (alpha* 1e-3)**2, (alpha*1e-3)**3,
                              (alpha*1e-3)**4, (alpha*1e-3)**5 ])
        return np.dot(coeff, alpha_pow) * 1e3
    
    def params2vel(self, x = None, alpha_max = None):
        """
        Convert parameters to a velocity model.
        """
        if x is None:
            x = self._model
        beta = x[:self._n_layers]
        alpha = beta * x[2*self._n_layers:]
        if alpha_max is not None:
            alpha = np.clip(alpha, 0., alpha_max)
        rho = self._nafe_drake(alpha)
        d = x[self._n_layers:2*self._n_layers]
        vel = np.hstack((alpha[:,None], beta[:,None], rho[:,None], d[:,None]))
        return vel
    
    def panel(self, nf = 200, th_kws = dict(ny = 200, domain = "fc", n_threads = 1)):
        """
        Compute 
        """
        faxis_full = np.unique(np.hstack([ f for f in self._faxis ]))
        faxis_new = np.linspace(faxis_full.min(), faxis_full.max(), nf)
        vel = self.params2vel()
        th = ThomsonHaskell(vel)
        th.propagate(faxis_new, **th_kws)
        return th
    
    def pick(self, modes = None, nf = 200, th_kws = dict(ny = 200, domain = "fc", n_threads = 1)):
        """
        """
        if modes is None:
            modes = self._modes
        th = self.panel()
        return th.pick(modes)
    
    def plot(self, nf = 100, ny = 100, figsize = (10, 5), cmap = None,
             n_threads = 1, th_kws = {}):
        """
        Plot results of inversion.
        
        Parameters
        ----------
        nf : int, default 100
            Number of samples for frequency axis.
        ny : int, default 100
            Number of samples for Y axis.
        figsize : tuple, default (10, 5)
            Figure width and height if axes is None.
        cmap : str, default "viridis"
            Colormap.
        n_threads : int, default 1
            Number of threads to pass to OpenMP for forward modelling.
        th_kws : dict
            Keyworded arguments passed to Thomson-Haskell plot.
        
        Returns
        -------
        ax1, ax2: matplotlib axes
            Axes used for plot.
        """
        if not hasattr(self, "_velocity_models"):
            raise ValueError("no inversion performed yet")
        if not isinstance(nf, int) or nf < 1:
            raise ValueError("nf must be a positive integer")
        if not isinstance(ny, int) or ny < 1:
            raise ValueError("ny must be a positive integer")
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError("figsize must be tuple with 2 elements")
        if cmap is None:
            cmap = self._set_cmap()
        else:
            if not hasattr(cm, cmap):
                raise ValueError("unknown cmap")
        if not isinstance(n_threads, int) or n_threads < 1:
            raise ValueError("n_threads must be a positive integer")
        if not isinstance(th_kws, dict):
            raise ValueError("th_kws must be a dictionary")
        
        faxis_full = np.unique(np.hstack([ f for f in self._faxis ]))
        faxis_new = np.linspace(faxis_full.min(), faxis_full.max(), nf)
        vel = self.best_velocity_model
        idx = np.argsort(1./self.misfits)
        misfits = 1./self.misfits
        misfits_normed = ( misfits - misfits.min() ) \
                         / ( misfits.max() - misfits.min() )
        colors = eval("cm.%s(misfits_normed)" % cmap)
        
        fig = plt.figure(figsize = figsize, facecolor = "white")
        fig.patch.set_alpha(0.)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        th = ThomsonHaskell(vel)
        vmin, vmax = max(0.1, np.floor(th._rayleigh_velocity().min())), vel[:,1].max()
        th.propagate(faxis_new, ny = ny, n_threads = n_threads)
        th.plot(axes = ax1, **th_kws)
        for f, c in zip(self._faxis, self._phase_velocity):
            ax1.plot(f, c, linewidth = 2, color = "red")
        ax1.set_xlim(faxis_new[0], faxis_new[-1])
        ax1.set_ylim(vmin, vmax)
        
        for i in idx:
            th = ThomsonHaskell(self._velocity_models[i])
            th.propagate(faxis_new, ny = ny, n_threads = n_threads)
            dc, fa = th.pick(self._modes)
            for c, f in zip(dc, fa):
                ax2.plot(f, c, linewidth = 2, color = colors[i])
        for f, c in zip(self._faxis, self._phase_velocity):
            ax2.plot(f, c, linewidth = 2, color = "red")
        ax2.set_xlabel("Frequency (Hz)", fontsize = 12)
        ax2.set_ylabel("Phase velocity (m/s)", fontsize = 12)
        ax2.set_xlim(faxis_new[0], faxis_new[-1])
        ax2.set_ylim(vmin, vmax)
        
        ax2.grid(True, linestyle = ":")
        fig.tight_layout()
        return ax1, ax2
    
    def plot_models(self, zmax, nz = 100, best = False, figsize = (10, 7),
                    color = None, color_best = None, plt_kws = dict(linewidth = 2)):
        """
        Plot inverted velocity models.
        
        Parameters
        ----------
        zmax : float
            Maximum depth (in m).
        nz : int
            Number of samples for depth.
        best : bool, default False
            Plot best model.
        figsize : tuple, default (10, 7)
            Figure width and height if axes is None.
        color : tuple or None, default None
            Mean model line color.
        color_best : tuple or None, default None
            Best model line color. Only used when best = True.
        plt_kws : dict
            Keyworded arguments passed to plot.
        
        Returns
        -------
        ax1, ax2: matplotlib axes
            Axes used for plot.
        """
        if not isinstance(zmax, (int, float)) or zmax < 0.:
            raise ValueError("zmax must be a positive float or integer")
        zmax_min = np.max([ d[:-1,3].sum() for d in self._velocity_models ])
        if zmax < zmax_min:
            raise ValueError("zmax must be greater than %.2f" % zmax_min)
        if not isinstance(nz, int) or nz < 1:
            raise ValueError("nz must be a positive integer")
        if not isinstance(best, bool):
            raise ValueError("best should be either True or False")
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError("figsize must be tuple with 2 elements")
        if color is not None:
            if not isinstance(color, tuple) or len(color) != 3 \
                or np.any([ c < 0. for c in color ]) or np.any([ c > 1. for c in color ]):
                raise ValueError("color must be a tuple with 3 elements between 0 and 1")
        else:
            color = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
        if best and color_best is not None:
            if not isinstance(color_best, tuple) or len(color_best) != 3 \
                or np.any([ c < 0. for c in color_best ]) or np.any([ c > 1. for c in color_best ]):
                raise ValueError("color_best must be a tuple with 3 elements between 0 and 1")
        else:
            color_best = (0.3333333333333333, 0.6588235294117647, 0.40784313725490196)
        if not isinstance(plt_kws, dict):
            raise ValueError("plt_kws must be a dictionary")
        
        lay = [ l for l in self._velocity_models ]
        vp = [ self._make_vel(l[:,0], l[:,3], zmax, nz) for l in lay ]
        vs = [ self._make_vel(l[:,1], l[:,3], zmax, nz) for l in lay ]
        
        idx = np.argsort(1./self.misfits)
        misfits = 1./self.misfits
        misfits_normed = ( misfits - misfits.min() ) \
                         / ( misfits.max() - misfits.min() )
        
        vp_mean = np.average(vp, 0, misfits_normed)
        vs_mean = np.average(vs, 0, misfits_normed)
        vp_std = np.std(vp, 0)
        vs_std = np.std(vs, 0)
        vp_low = np.mean(vp, 0) - vp_std
        vp_high = np.mean(vp, 0) + vp_std
        vs_low = np.mean(vs, 0) - vs_std
        vs_high = np.mean(vs, 0) + vs_std
        
        fig = plt.figure(figsize = figsize, facecolor = "white")
        fig.patch.set_alpha(0.)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        az = np.linspace(0., zmax, nz)
        ax1.fill_betweenx(az, vs_low, vs_high, color = color, alpha = 0.15)
        ax1.plot(vs_mean, az, color = color, **plt_kws)
        ax2.fill_betweenx(az, vp_low, vp_high, color = color, alpha = 0.15)
        ax2.plot(vp_mean, az, color = color, **plt_kws)
        if best:
            vp_best = vp[idx[-1]]
            vs_best = vs[idx[-1]]
            ax1.plot(vs_best, az, linewidth = 2, color = color_best)
            ax2.plot(vp_best, az, linewidth = 2, color = color_best)
        ax1.set_xlabel("S-wave velocity (m/s)", fontsize = 12)
        ax1.set_ylabel("Depth (m)", fontsize = 12)
        ax2.set_xlabel("P-wave velocity (m/s)", fontsize = 12)
        ax2.set_ylabel("Depth (m)", fontsize = 12)
        ax1.set_ylim(0., zmax)
        ax2.set_ylim(0., zmax)
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        
        ax1.grid(True, linestyle = ":")
        ax2.grid(True, linestyle = ":")
        fig.tight_layout()
        return ax1, ax2
    
    def _set_cmap(self):
        if hasattr(cm, "viridis"):
            return "viridis"
        else:
            return "jet"
    
    def _make_vel(self, lay, thick, zmax, nz):
        zint = np.array(thick)
        zint = zint.cumsum()
        zint[-1] = zmax
        dz = zmax / nz
        return lay2vel(np.hstack((lay[:,None], zint[:,None])), dz, (nz,))
    
    def save(self, filename):
        """
        Pickle the dispersion curve to a file.
        
        Parameters
        ----------
        filename: str
            Pickle filename.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol = pickle.HIGHEST_PROTOCOL)
    
    @property
    def phase_velocity(self):
        """
        list of ndarray
        Observed phase velocities (in m/s). The ith element corresponds to the
        ith mode in modes.
        """
        return self._phase_velocity
    
    @phase_velocity.setter
    def phase_velocity(self, value):
        self._phase_velocity = value
        
    @property
    def faxis(self):
        """
        list of ndarray
        Frequency axis (in Hz). The ith element corresponds to the ith mode
        in modes.
        """
        return self._faxis
    
    @faxis.setter
    def faxis(self, value):
        self._faxis = value
        
    @property
    def modes(self):
        """
        list of int
        Mode numbers associated to elements in phase_velocity in faxis.
        """
        return self._modes
    
    @modes.setter
    def modes(self, value):
        self._modes = value
    
    @property
    def model(self):
        if hasattr(self, "_model"):
            return self._model
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def misfit(self):
        if hasattr(self, "_misfit"):
            return self._misfit
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def models(self):
        if hasattr(self, "_models"):
            return self._models
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def misfits(self):
        if hasattr(self, "_misfits"):
            return self._misfits
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def energy(self):
        energy = np.min(self.misfits, axis = 0)
        return np.array([ np.min(energy[:i+1]) for i in range(len(energy)) ])
            
    @property
    def n_iter(self):
        if hasattr(self, "_n_iter"):
            return self._n_iter
        else:
            raise AttributeError("no inversion performed yet")
            
    @property
    def n_eval(self):
        if hasattr(self, "_n_eval"):
            return self._n_eval
        else:
            raise AttributeError("no inversion performed yet")