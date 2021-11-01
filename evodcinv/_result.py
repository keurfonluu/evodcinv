import numpy

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter

from disba import surf96
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
        perc=0.99,
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
            # Filter and sort models
            models, misfits = self._filter_results(perc)
            idx = numpy.argsort(misfits)[::-1]
            models = models[idx]
            misfits = misfits[idx]

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

    def _filter_results(self, perc=0.99):
        apost = numpy.exp(-0.5 * self.misfits ** 2)
        threshold = perc * apost.max()
        idx = apost > threshold
        
        return self.models[idx], self.misfits[idx]

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
