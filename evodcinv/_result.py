import matplotlib.pyplot as plt
import numpy
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
                global_misfits=numpy.row_stack(
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

        def profile(thickness, parameter, z):
            zp, fp = resample(thickness, parameter, dz)
            zp = zp.cumsum()

            return numpy.interp(z, zp, fp)

        models = self.models
        misfits = self.misfits

        if zmax is None:
            zmax = numpy.max([model[:-1, 0].sum() for model in models])
        nz = numpy.ceil(zmax / dz).astype(int)
        z = dz * numpy.arange(nz)

        mean_model = numpy.column_stack([
            numpy.average(
                [profile(model[:, 0], model[:, i + 1], z) for model in models],
                axis=0,
                weights=1.0 / misfits,
            )
            for i in range(3)
            ]
        )
        d = numpy.full_like(z, dz)

        return numpy.column_stack((d, mean_model))

    def plot_curve(
        self,
        period,
        mode,
        wave,
        type,
        show="best",
        disba_args=None,
        plot_args=None,
        ax=None,
    ):
        assert type in {"phase", "group", "ellipticity"}
        assert show in {"best", "all"}

        # disba arguments
        disba_args = disba_args if disba_args is not None else {}
        _disba_args = {
            "algorithm": "dunkin",
            "dc": 0.001,
            "dt": 0.01,
        }
        _disba_args.update(disba_args)

        algorithm = _disba_args.pop("algorithm")
        dc = _disba_args.pop("dc")
        dt = _disba_args.pop("dt")

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
                    ifunc[algorithm][wave],
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
                    algorithm,
                    dc,
                )
                rel = ell(period, mode)

                return numpy.abs(rel.ellipticity)

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

        assert plot_type in {"line", "loglog", "semilogx", "semilogy"}
        assert xaxis in {"frequency", "period"}
        assert yaxis in {"slowness", "velocity"}

        plot_type = plot_type if plot_type != "line" else "plot"
        plot = getattr(plt if ax is None else ax, plot_type)
        x = 1.0 / period if xaxis == "frequency" else period

        if show == "all":
            # Sort models
            idx = numpy.argsort(self.misfits)[::-1]
            models = self.models[idx]
            misfits = self.misfits[idx]

            # Make colormap
            norm = Normalize(misfits.min(), misfits.max())
            smap = ScalarMappable(norm, cmap)
            smap.set_array([])

            # Generate and plot curves
            curves = [get_y(*model.T) for model in models]
            for curve, misfit in zip(curves, misfits):
                y = (
                    1.0 / curve
                    if "type" != "ellipticity" and yaxis == "slowness"
                    else curve
                )
                plot(x, y, color=smap.to_rgba(misfit), **_plot_args)

        elif show == "best":
            curve = get_y(*self.model.T)
            y = (
                1.0 / curve
                if "type" != "ellipticity" and yaxis == "slowness"
                else curve
            )
            plot(x, y, **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        xlabel = f"{xaxis.capitalize()} [{units[xaxis]}]"
        ylabel = f"{type.capitalize()} "
        ylabel += "[H/V]" if type == "ellipticity" else f"{yaxis} [km/s]"
        gca.set_xlabel(xlabel)
        gca.set_ylabel(ylabel)

        # gca.set_xlim(x.min(), x.max())

        # Disable exponential tick labels
        gca.xaxis.set_major_formatter(ScalarFormatter())
        gca.xaxis.set_minor_formatter(ScalarFormatter())

    def plot_model(self, parameter, zmax=None, show="best", dz=None, plot_args=None, ax=None):
        parameters = {
            "velocity_p": 1,
            "velocity_s": 2,
            "density": 3,
            "vp": 1,
            "vs": 2,
            "rho": 3,
        }
        assert parameter in parameters
        assert show in {"best", "mean", "all"}
        if show == "mean":
            assert dz is not None

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
        maxiter = self.maxiter if self.n_runs > 1 else [self.maxiter]
        global_misfits = self.global_misfits if self.n_runs > 1 else [self.global_misfits]

        assert run in {"all"} or (isinstance(run, int) and run > 0)
        if isinstance(run, int) and run > 1:
            assert run <= self.n_runs

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
                plot(numpy.arange(n) + 1, misfits, **_plot_args)

        else:
            plot(numpy.arange(maxiter[run - 1]) + 1, global_misfits[run - 1], **_plot_args)

        # Customize axes
        gca = ax if ax is not None else plt.gca()

        gca.set_xlabel("Iteration")
        gca.set_ylabel("Misfit value")

        xmax = numpy.max(maxiter) if run == "all" else maxiter[run - 1]
        gca.set_xlim(1, xmax)

    def threshold(self, value):
        idx = self.misfits <= value

        return InversionResult(
            xs=self.xs,
            models=self.models[idx],
            misfits=self.misfits[idx],
            global_misfits=self.global_misfits,
            maxiter=self.maxiter,
            popsize=self.popsize,
        )

    def write(self, filename, file_format=None, **kwargs):
        from ._io import write

        write(filename, self, file_format, **kwargs)

    @property
    def misfit(self):
        return self.misfits.min()

    @property
    def x(self):
        return self.xs[self.misfits.argmin()]

    @property
    def model(self):
        return self.models[self.misfits.argmin()]

    @property
    def n_runs(self):
        return (
            len(self.maxiter)
            if numpy.ndim(self.maxiter)
            else 1
        )
