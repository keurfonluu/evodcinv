import numpy

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

from disba import surf96, depthplot
from disba._common import ifunc

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

            return "\n".join([
                "misfit".rjust(m) + ": " + repr(self.misfit),
                "x".rjust(m) + ": " + repr(self.x),
                "model".rjust(m) + ": " + repr(self.model),
            ])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Return a list of attributes."""
        return list(self.keys())

    def __add__(self, other):
        assert isinstance(other, InversionResult)

        if not self.keys():
            return InversionResult(**other)

        elif not other.keys():
            return InversionResult(**self)

        else:
            return InversionResult(
                xs=numpy.vstack([self.xs, other.xs]),
                models=numpy.vstack([self.models, other.models]),
                misfits=numpy.concatenate([self.misfits, other.misfits]),
                global_misfits=numpy.row_stack((self.global_misfits, other.global_misfits)),
                maxiter=([self.maxiter] if isinstance(self.maxiter, int) else self.maxiter) + ([other.maxiter] if isinstance(other.maxiter, int) else other.maxiter),
                popsize=([self.popsize] if isinstance(self.popsize, int) else self.popsize) + ([other.popsize] if isinstance(other.popsize, int) else other.popsize),
            )

    def plot_curve(
        self,
        t,
        mode,
        wave,
        type,
        all=False,
        disba_args=None,
        plot_args=None,
        ax=None,
    ):
        # disba arguments
        disba_args = disba_args if disba_args is not None else {}
        _disba_args = {
            "algorithm": "dunkin",
            "dc": 0.001,
            "dt": 0.025,
        }
        _disba_args.update(disba_args)

        algorithm = _disba_args.pop("algorithm")
        dc = _disba_args.pop("dc")
        dt = _disba_args.pop("dt")

        def getc(thickness, velocity_p, velocity_s, density):
            c = surf96(
                t,
                thickness,
                velocity_p,
                velocity_s,
                density,
                mode,
                itype[type],
                ifunc[algorithm][wave],
                dc,
                dt,
            )
            idx = c > 0.0
            
            return c[idx]

        # Plot arguments
        plot_args = plot_args if plot_args is not None else {}
        _plot_args = {
            "type": "line",
            "xaxis": "period",
            "cmap": "viridis_r",
        }
        _plot_args.update(plot_args)

        plot_type = _plot_args.pop("type")
        xaxis = _plot_args.pop("xaxis")
        cmap = _plot_args.pop("cmap")

        assert plot_type in {"line", "loglog", "semilogx", "semilogy"}
        assert xaxis in {"frequency", "period"}

        plot_type = plot_type if plot_type != "line" else "plot"
        plot = getattr(plt if ax is None else ax, plot_type)
        x = 1.0 / t if xaxis == "frequency" else t

        if all:
            # Sort models
            idx = numpy.argsort(self.misfits)[::-1]
            models = self.models[idx]
            misfits = self.misfits[idx]

            # Make colormap
            norm = Normalize(misfits.min(), misfits.max())
            smap = ScalarMappable(norm, cmap)
            smap.set_array([])

            # Generate and plot curves
            curves = [getc(*model.T) for model in models]
            for curve, misfit in zip(curves, misfits):
                plot(x, curve, color=smap.to_rgba(misfit), **_plot_args)

        else:
            c = getc(*self.model.T)
            plot(x, c, **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        xlabel = f"{xaxis.capitalize()} [{units[xaxis]}]"
        ylabel = f"{type.capitalize()} velocity [km/s]"
        gca.set_xlabel(xlabel)
        gca.set_ylabel(ylabel)

        gca.set_xlim(x.min(), x.max())

        # Disable exponential tick labels
        gca.xaxis.set_major_formatter(ScalarFormatter())
        gca.xaxis.set_minor_formatter(ScalarFormatter())

    def plot_model(
        self,
        parameter,
        zmax=None,
        all=False,
        plot_args=None,
        ax=None
    ):
        parameters = {
            "velocity_p": 1,
            "velocity_s": 2,
            "density": 3,
            "vp": 1,
            "vs": 2,
            "rho": 3,
        }
        assert parameter in parameters

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
        plot = getattr(plt if ax is None else ax, "plot")
        i = parameters[parameter]

        if all:
            # Sort models
            idx = numpy.argsort(self.misfits)[::-1]
            models = self.models[idx]
            misfits = self.misfits[idx]

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
            model = self.model
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

    def plot_misfit(self, plot_args=None, ax=None):
        # Plot arguments
        plot_args = plot_args if plot_args is not None else {}
        _plot_args = {
            "color": "black",
            "linewidth": 2,
        }
        _plot_args.update(plot_args)

        plot = getattr(plt if ax is None else ax, "plot")
        plot(numpy.arange(self.maxiter) + 1, self.global_misfits, **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        gca.set_xlabel("Iteration")
        gca.set_ylabel("Misfit")

        gca.set_xlim(1, self.maxiter)

    def threshold(self, perc=0.99):
        apost = numpy.exp(-0.5 * self.misfits ** 2)
        threshold = perc * apost.max()
        idx = apost > threshold

        return InversionResult(
            xs=self.xs,
            models=self.models[idx],
            misfits=self.misfits[idx],
            global_misfits=self.global_misfits,
            maxiter=self.maxiter,
            popsize=self.popsize,
        )

    def write(self, filename, indent=None):
        from ._io import write

        write(filename, self, indent)

    @property
    def misfit(self):
        return self.misfits.min()

    @property
    def x(self):
        return self.xs[self.misfits.argmin()]

    @property
    def model(self):
        return self.models[self.misfits.argmin()]
