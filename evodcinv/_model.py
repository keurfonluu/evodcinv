from collections import namedtuple

import numpy
from disba import DispersionError, Ellipticity, surf96
from disba._common import ifunc
from disba._helpers import is_sorted
from stochopy.optimize import minimize

from ._common import itype
from ._helpers import get_velocity_p, nafe_drake
from ._result import InversionResult
from ._progress import ProgressBar

Layer = namedtuple("Layer", ["thickness", "velocity_s", "poisson"])
Curve = namedtuple("Curve", ["period", "data", "mode", "wave", "type", "weight", "uncertainties"])


constraints = {
    "cpso": "Shrink",
    "pso": "Shrink",
    "de": "Random",
    "cmaes": "Penalize",
    "vdcma": "Penalize",
}


class EarthModel:
    def __init__(self):
        self._layers = []
        self._curves = []
        self._extra_terms = []

        self.set_misfit_func("rmse")
        self.set_density_func("nafe-drake")

    def add_layer(self, thickness, velocity_s, poisson=[0.2, 0.4]):
        assert len(thickness) == 2
        assert len(velocity_s) == 2
        assert len(poisson) == 2

        assert thickness[0] <= thickness[1]
        assert velocity_s[0] <= velocity_s[1]
        assert poisson[0] <= poisson[1]

        self._layers.append(Layer(tuple(thickness), tuple(velocity_s), tuple(poisson)))

    def add_curve(self, period, data, mode, wave, type, weight=1.0, uncertainties=None):
        assert len(period) == len(data)
        assert mode >= 0
        assert wave in {"rayleigh", "love"}
        assert type in {"phase", "group", "ellipticity"}
        assert is_sorted(period)

        if uncertainties is not None:
            if isinstance(uncertainties, (int, float)):
                uncertainties = numpy.full_like(data, uncertainties)

            elif numpy.ndim(uncertainties) == 1:
                assert len(uncertainties) == len(data)
                uncertainties = numpy.asarray(uncertainties)

            else:
                raise ValueError()

        self._curves.append(
            Curve(numpy.asarray(period), numpy.asarray(data), mode, wave, type, weight, uncertainties)
        )

    def add_misfit_term(self, func):
        self._extra_terms.append(func)

    def set_misfit_func(self, func):
        if func == "norm1":
            self._misfit_func = lambda x: numpy.abs(x).sum()

        elif func == "norm2":
            self._misfit_func = lambda x: numpy.square(x).sum()

        elif func == "rmse":
            self._misfit_func = lambda x: (numpy.square(x).sum() / len(x)) ** 0.5

        elif hasattr(func, "__call__"):
            self._misfit_func = func

        else:
            raise ValueError()

    def set_density_func(self, func):
        if func == "nafe-drake":
            self._density_func = nafe_drake

        elif hasattr(func, "__call__"):
            self._density_func = func

        else:
            raise ValueError()

    def invert(self, algorithm="dunkin", dc=0.001, dt=0.01, optimizer_args=None):
        # Optimizer arguments
        optimizer_args = optimizer_args if optimizer_args is not None else {}
        _optimizer_args = {
            "method": "cpso",
            "maxiter": 100,
            "popsize": 10,
        }
        _optimizer_args.update(optimizer_args)

        method = _optimizer_args.pop("method")

        # Overwrite options
        _optimizer_args["return_all"] = True
        if method != "na":
            _optimizer_args["constraints"] = constraints[method]

        # Minimize misfit function
        func = lambda x: self._misfit_function(x, algorithm, dc, dt)
        bounds = numpy.vstack(
            [
                [layer.thickness for layer in self._layers[:-1]],
                [layer.velocity_s for layer in self._layers],
                [layer.poisson for layer in self._layers],
            ]
        )

        with ProgressBar(max=_optimizer_args["maxiter"]) as bar:
            def callback(X, res):
                bar.misfit = res.fun
                bar.next()

            x = minimize(func, bounds, method=method, options=_optimizer_args, callback=callback)

        # Parse output
        maxiter = _optimizer_args["maxiter"]
        popsize = _optimizer_args["popsize"]

        velocity_models = numpy.empty((maxiter, popsize, self.n_layers, 4))
        for i in range(maxiter):
            for j in range(popsize):
                velocity_models[i, j] = numpy.transpose(
                    self.transform(x.xall[i, j])
                )

        out = InversionResult(
            xs=numpy.concatenate(x.xall),
            models=numpy.concatenate(velocity_models),
            misfits=numpy.concatenate(x.funall),
            global_misfits=numpy.minimum.accumulate(x.funall.min(axis=1)),
            maxiter=maxiter,
            popsize=popsize,
        )

        return out

    def transform(self, x):
        thickness = x[: self.n_layers - 1]
        velocity_s = x[self.n_layers - 1 : 2 * self.n_layers - 1]
        poisson = x[2 * self.n_layers - 1 :]
        velocity_p = get_velocity_p(velocity_s, poisson)
        density = self._get_density(velocity_p)

        return numpy.append(thickness, 1.0), velocity_p, velocity_s, density

    def _misfit_function(self, x, algorithm, dc, dt):
        thickness, velocity_p, velocity_s, density = self.transform(x)

        error_extra = 0.0
        if len(self._extra_terms):
            for extra_term in self._extra_terms:
                error_extra += extra_term(x)

            if numpy.isinf(error_extra):
                return numpy.Inf

        error = 0.0
        weights_sum = 0.0
        for curve in self._curves:
            try:
                if curve.type != "ellipticity":
                    c = surf96(
                        curve.period,
                        thickness,
                        velocity_p,
                        velocity_s,
                        density,
                        curve.mode,
                        itype[curve.type],
                        ifunc[algorithm][curve.wave],
                        dc,
                        dt,
                    )
                    idx = c > 0.0
                    dcalc = c[idx]

                else:
                    ell = Ellipticity(
                        thickness,
                        velocity_p,
                        velocity_s,
                        density,
                        algorithm,
                        dc,
                    )
                    rel = ell(curve.period, mode=curve.mode)
                    dcalc = numpy.abs(rel.ellipticity)

                n = len(dcalc)
                sigma = (
                    curve.data[:n]
                    if curve.uncertainties is None
                    else curve.uncertainties[:n]
                )
                error += curve.weight * self._misfit_func((dcalc - curve.data[:n]) / sigma)
                weights_sum += curve.weight

            except DispersionError:
                return numpy.Inf

        return error / weights_sum + error_extra

    def _get_density(self, velocity_p):
        return numpy.array([self._density_func(vp) for vp in velocity_p])

    @property
    def layers(self):
        return self._layers

    @property
    def curves(self):
        return self._curves

    @property
    def extra_terms(self):
        return self._extra_terms

    @property
    def n_layers(self):
        return len(self._layers)

    @property
    def n_curves(self):
        return len(self._curves)

    @property
    def n_extra_terms(self):
        return len(self._extra_terms)
