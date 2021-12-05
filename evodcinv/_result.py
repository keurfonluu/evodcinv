import matplotlib.pyplot as plt
import numpy as np
from disba import depthplot, resample, surf96, Ellipticity
from disba._common import ifunc
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

from ._common import itype, units


class InversionResult(dict):
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        """Define dict.attr as an alias of dict[attr]."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __repr__(self):
        """Pretty result."""
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1

            return "\n".join(
                [
                    "misfit".rjust(m) + ": " + repr(self.misfit),
                    "x".rjust(m) + ": " + repr(self.x),
                    "model".rjust(m) + ": " + repr(self.model),
                ]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Return a list of attributes."""
        return list(self.keys())

    def __len__(self):
        """Return number of models."""
        return len(self.models)

    def __add__(self, other):
        """Concatenate inversion results."""
        if not isinstance(other, InversionResult):
            raise TypeError()

        if not self.keys():
            return InversionResult(**other)

        elif not other.keys():
            return InversionResult(**self)

        else:
            return InversionResult(
                xs=np.vstack([self.xs, other.xs]),
                models=np.vstack([self.models, other.models]),
                misfits=np.concatenate([self.misfits, other.misfits]),
                global_misfits=np.row_stack(
                    (self.global_misfits, other.global_misfits)
                ),
                maxiter=(
                    [self.maxiter] if isinstance(self.maxiter, int) else self.maxiter
                )
                + (
                    [other.maxiter] if isinstance(other.maxiter, int) else other.maxiter
                ),
                popsize=(
                    [self.popsize] if isinstance(self.popsize, int) else self.popsize
                )
                + (
                    [other.popsize] if isinstance(other.popsize, int) else other.popsize
                ),
            )

    def mean(self, dz, zmax=None):
        """
        Calculate mean velocity model.

        Parameters
        ----------
        dz : scalar
            Maximum layer thickness (in km).
        zmax : scalar or None, optional, default None
            Depth of last data point.
        
        """

        def profile(thickness, parameter, z):
            """Interpolate along model depth."""
            zp, fp = resample(thickness, parameter, dz)
            zp = zp.cumsum()

            return np.interp(z, zp, fp)

        models = self.models
        misfits = self.misfits

        if zmax is None:
            zmax = np.max([model[:-1, 0].sum() for model in models])
        nz = np.ceil(zmax / dz).astype(int)
        z = dz * np.arange(nz)

        mean_model = np.column_stack([
            np.average(
                [profile(model[:, 0], model[:, i + 1], z) for model in models],
                axis=0,
                weights=1.0 / misfits,
            )
            for i in range(3)
            ]
        )
        d = np.full_like(z, dz)

        return np.column_stack((d, mean_model))

    def plot_curve(
        self,
        period,
        mode,
        wave,
        type,
        show="best",
        stride=1,
        n_jobs=-1,
        dc=0.001,
        dt=0.01,
        plot_args=None,
        ax=None,
    ):
        """
        Plot calculated data curves.

        Parameters
        ----------
        period : array_like
            Periods (in s).
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.
        type : str {'phase', 'group', 'ellipticity'}, optional, default 'phase'
            Data type.
        show : str {'best', 'all'}, optional, default 'best'
            Model to use to calculate data curves.
        stride : int, optional, default 1
            Number of models to skip.
        n_jobs : int, optional, default -1
            Number of CPU cores to calculate data curves in parallel. Supply -1 to use all available CPU cores. Only used if ``show = "all"``.
        dc : scalar, optional, default 0.001
            Phase velocity increment for root finding.
        dt : scalar, optional, default 0.01
            Frequency increment (%) for calculating group velocity.
        plot_args : dict or None, optional, default None
            A dictionary of options to pass to plotting function.
        ax : :class:`matplotlib.pyplot.Axes` or None, optional, default None
            Matplotlib axes. If `None`, use current axes.
        
        """
        from joblib import Parallel, delayed

        if type not in {"phase", "group", "ellipticity"}:
            raise ValueError()
        if show not in {"best", "all"}:
            raise ValueError()

        if type in {"phase", "group"}:
            def get_y(thickness, velocity_p, velocity_s, density):
                c = surf96(
                    period,
                    thickness,
                    velocity_p,
                    velocity_s,
                    density,
                    mode,
                    itype[type],
                    ifunc["dunkin"][wave],
                    dc,
                    dt,
                )
                idx = c > 0.0

                return c[idx]

        else:
            def get_y(thickness, velocity_p, velocity_s, density):
                ell = Ellipticity(
                    thickness,
                    velocity_p,
                    velocity_s,
                    density,
                    "dunkin",
                    dc,
                )
                rel = ell(period, mode)

                return np.abs(rel.ellipticity)

        # Plot arguments
        plot_args = plot_args if plot_args is not None else {}
        _plot_args = {
            "type": "line",
            "xaxis": "period",
            "yaxis": "velocity",
            "cmap": "viridis_r",
        }
        _plot_args.update(plot_args)

        plot_type = _plot_args.pop("type")
        xaxis = _plot_args.pop("xaxis")
        yaxis = _plot_args.pop("yaxis")
        cmap = _plot_args.pop("cmap")

        if plot_type not in {"line", "loglog", "semilogx", "semilogy"}:
            raise ValueError()
        if xaxis not in {"frequency", "period"}:
            raise ValueError()
        if yaxis not in {"slowness", "velocity"}:
            raise ValueError()

        plot_type = plot_type if plot_type != "line" else "plot"
        plot = getattr(plt if ax is None else ax, plot_type)
        x = 1.0 / period if xaxis == "frequency" else period

        if show == "all":
            # Sort models
            idx = np.argsort(self.misfits)[::-1]
            models = self.models[idx]
            misfits = self.misfits[idx]

            # Skip models
            models = models[::stride]
            misfits = misfits[::stride]

            # Make colormap
            norm = Normalize(misfits.min(), misfits.max())
            smap = ScalarMappable(norm, cmap)
            smap.set_array([])

            # Generate and plot curves
            curves = Parallel(n_jobs=n_jobs)(delayed(get_y)(*model.T) for model in models)
            for curve, misfit in zip(curves, misfits):
                y = (
                    1.0 / curve
                    if "type" != "ellipticity" and yaxis == "slowness"
                    else curve
                )
                plot(x[:len(y)], y, color=smap.to_rgba(misfit), **_plot_args)

        elif show == "best":
            curve = get_y(*self.model.T)
            y = (
                1.0 / curve
                if "type" != "ellipticity" and yaxis == "slowness"
                else curve
            )
            plot(x[:len(y)], y, **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        xlabel = f"{xaxis.capitalize()} [{units[xaxis]}]"
        ylabel = f"{type.capitalize()} "
        ylabel += "[H/V]" if type == "ellipticity" else f"{yaxis} [km/s]"
        gca.set_xlabel(xlabel)
        gca.set_ylabel(ylabel)

        # Disable exponential tick labels
        gca.xaxis.set_major_formatter(ScalarFormatter())
        gca.xaxis.set_minor_formatter(ScalarFormatter())

    def plot_model(self, parameter, zmax=None, show="best", stride=1, dz=None, plot_args=None, ax=None):
        """
        Plot model parameter as a function of depth.

        Parameters
        ----------
        parameter : str
            Parameter to plot. Should be one of:

             - 'velocity_p' or 'vp'
             - 'velocity_s' or 'vs'
             - 'density' or 'rho'

        zmax : scalar or None, optional, default None
            Depth of last data point.
        show : str {'best', 'all'}, optional, default 'best'
            Model to plot.
        stride : int, optional, default 1
            Number of models to skip.
        dz : scalar or None, optional, default None
            Maximum layer thickness (in km).
        plot_args : dict or None, optional, default None
            A dictionary of options to pass to plotting function.
        ax : :class:`matplotlib.pyplot.Axes` or None, optional, default None
            Matplotlib axes. If `None`, use current axes.

        """
        parameters = {
            "velocity_p": 1,
            "velocity_s": 2,
            "density": 3,
            "vp": 1,
            "vs": 2,
            "rho": 3,
        }
        if parameter not in parameters:
            raise ValueError()
        if show not in {"best", "mean", "all"}:
            raise ValueError()
        if show == "mean" and dz is None:
            raise ValueError()

        # Plot arguments
        plot_args = plot_args if plot_args is not None else {}
        _plot_args = {
            "cmap": "viridis_r",
            "color": "black",
            "linewidth": 2,
        }
        _plot_args.update(plot_args)

        cmap = _plot_args.pop("cmap")

        # Plot
        i = parameters[parameter]

        if show == "all":
            # Sort models
            idx = np.argsort(self.misfits)[::-1]
            models = self.models[idx]
            misfits = self.misfits[idx]

            # Skip models
            models = models[::stride]
            misfits = misfits[::stride]

            # Make colormap
            norm = Normalize(misfits.min(), misfits.max())
            smap = ScalarMappable(norm, cmap)
            smap.set_array([])

            # Generate and plot curves
            for model, misfit in zip(models, misfits):
                tmp = {k: v for k, v in _plot_args.items()}
                tmp["color"] = smap.to_rgba(misfit)
                depthplot(model[:, 0], model[:, i], zmax, plot_args=tmp, ax=ax)

        else:
            model = (
                self.model
                if show == "best"
                else self.mean(dz, zmax)
            )
            depthplot(model[:, 0], model[:, i], zmax, plot_args=_plot_args, ax=ax)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        labels = {
            "velocity_p": "P-wave velocity [km/s]",
            "velocity_s": "S-wave velocity [km/s]",
            "density": "Density [$g/cm^3$]",
            "vp": "$V_p$ [km/s]",
            "vs": "$V_s$ [km/s]",
            "rho": "$\\rho$ [$g/cm^3$]",
        }

        xlabel = labels[parameter]
        ylabel = "Depth [km]"
        gca.set_xlabel(xlabel)
        gca.set_ylabel(ylabel)

    def plot_misfit(self, run="all", plot_args=None, ax=None):
        """
        Plot misfit as a function of iteration number.

        Parameters
        ----------
        run : int or str {'all'}, optional, default 'all'
            Run for which misfit to be plot.
        plot_args : dict or None, optional, default None
            A dictionary of options to pass to plotting function.
        ax : :class:`matplotlib.pyplot.Axes` or None, optional, default None
            Matplotlib axes. If `None`, use current axes.
        
        """
        maxiter = self.maxiter if self.n_runs > 1 else [self.maxiter]
        global_misfits = self.global_misfits if self.n_runs > 1 else [self.global_misfits]

        if not (run in {"all"} or (isinstance(run, int) and run > 0)):
            raise ValueError()
        if isinstance(run, int) and run > 1:
            if run > self.n_runs:
                raise ValueError()

        # Plot arguments
        plot_args = plot_args if plot_args is not None else {}
        _plot_args = {
            "color": "black",
            "linewidth": 2,
        }
        _plot_args.update(plot_args)

        plot = getattr(plt if ax is None else ax, "plot")
        if run == "all":
            for n, misfits in zip(maxiter, global_misfits):
                plot(np.arange(n) + 1, misfits, **_plot_args)

        else:
            plot(np.arange(maxiter[run - 1]) + 1, global_misfits[run - 1], **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        gca.set_xlabel("Iteration")
        gca.set_ylabel("Misfit value")

        xmax = np.max(maxiter) if run == "all" else maxiter[run - 1]
        gca.set_xlim(1, xmax)

    def threshold(self, value=None):
        """
        Apply a threshold filter.

        Remove models that do not satisfy the threshold criterion.

        Parameters
        ----------
        value : scalar or None, optional, default None
            Single value to be used for the data threshold. If None, remove invalid models (i.e., with misfit equal to Inf).
        
        Returns
        -------
        :class:`evodcinv.InversionResult`
            Inversion results with models that satisfy the threshold criterion.

        """
        idx = (
            ~np.isinf(self.misfits)
            if value is None
            else self.misfits <= value
        )

        return InversionResult(
            xs=self.xs,
            models=self.models[idx],
            misfits=self.misfits[idx],
            global_misfits=self.global_misfits,
            maxiter=self.maxiter,
            popsize=self.popsize,
        )

    def write(self, filename, file_format=None, **kwargs):
        """
        Write inversion results to a file.

        Parameters
        ----------
        filename : str
            Output file name.
        file_format : str {'h5', 'json'} or None, optional, default None
            Output file format.
        
        """
        from ._io import write

        write(filename, self, file_format, **kwargs)

    @property
    def misfit(self):
        """Return best fit model misfit."""
        return self.misfits.min()

    @property
    def x(self):
        """Return best fit model parameters."""
        return self.xs[self.misfits.argmin()]

    @property
    def model(self):
        """Return best fit model."""
        return self.models[self.misfits.argmin()]

    @property
    def n_runs(self):
        """Return number of runs."""
        return (
            len(self.maxiter)
            if np.ndim(self.maxiter)
            else 1
        )
