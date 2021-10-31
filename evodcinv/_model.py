from collections import namedtuple

import numpy

from disba import DispersionError, Ellipticity, surf96
from disba._common import ifunc

from stochopy.optimize import minimize

from ._helpers import InversionSummary, get_velocity_p, nafe_drake


Layer = namedtuple("Layer", ["thickness", "velocity_s", "poisson"])
Curve = namedtuple("Curve", ["period", "data", "mode", "wave", "type", "weight"])

itype = {
    "phase": 0,
    "group": 1,
}

constraints = {
    "cpso": "Shrink",
    "pso": "Shrink",
    "de": "Random",
    "cmaes": "Penalize",
    "vdcma": "Penalize",
}


class EarthModel:
    def __init__(self, algorithm="dunkin", dc=0.005, dt=0.025):
        self._algorithm = algorithm
        self._dc = dc
        self._dt = dt

        self._layers = []
        self._curves = []
        self._density_func = nafe_drake

    def add_layer(self, thickness, velocity_s, poisson=[0.2, 0.4]):
        assert len(thickness) == 2
        assert len(velocity_s) == 2
        assert len(poisson) == 2

        assert thickness[0] <= thickness[1]
        assert velocity_s[0] <= velocity_s[1]
        assert poisson[0] <= poisson[1]

        self._layers.append(Layer(tuple(thickness), tuple(velocity_s), tuple(poisson)))

    def add_curve(self, period, data, mode, wave, type, weight=1.0):
        assert len(period) == len(data)
        assert mode >= 0
        assert wave in {"rayleigh", "love"}
        assert type in {"phase", "group", "ellipticity"}

        self._curves.append(Curve(numpy.asarray(period), numpy.asarray(data), mode, wave, type, weight))

    def set_density_func(self, func):
        self._density_func = func

    def invert(self, method="cpso", options=None):
        options = options if options is not None else {}

        # Default options
        _options = {
            "maxiter": 100,
            "popsize": 10,
        }
        _options.update(options)

        # Overwrite options
        _options["return_all"] = True
        _options["constraints"] = constraints[method]

        # Minimize misfit function
        func = self._misfit_function
        bounds = numpy.vstack([
            [layer.thickness for layer in self._layers],
            [layer.velocity_s for layer in self._layers],
            [layer.poisson for layer in self._layers],
        ])
        x = minimize(func, bounds, method=method, options=_options)

        # Parse output
        maxiter = _options["maxiter"]
        popsize = _options["popsize"]

        velocity_models = numpy.empty((maxiter, popsize, self.n_layers, 4))
        for i in range(maxiter):
            for j in range(popsize):
                velocity_models[i, j] = numpy.transpose(self._parse_parameters(x.xall[i, j]))

        out = InversionSummary(
            misfit=x.fun,
            x=x.x,
            model=numpy.transpose(self._parse_parameters(x.x)),
            all_x=numpy.concatenate(x.xall),
            all_models=numpy.concatenate(velocity_models),
            all_misfits=numpy.concatenate(x.funall),
            global_misfits=numpy.minimum.accumulate(x.funall.min(axis=1)),
            maxiter=maxiter,
            popsize=popsize,
        )

        return out

    def _parse_parameters(self, x):
        thickness = x[:self.n_layers]
        velocity_s = x[self.n_layers : 2 * self.n_layers]
        poisson = x[2 * self.n_layers:]
        velocity_p = get_velocity_p(velocity_s, poisson)
        density = self._get_density(velocity_p)

        return thickness, velocity_p, velocity_s, density

    def _misfit_function(self, x):
        thickness, velocity_p, velocity_s, density = self._parse_parameters(x)

        error = 0.0
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
                        ifunc[self._algorithm][curve.wave],
                        self._dc,
                    )
                    idx = c > 0.0
                    dcalc = c[idx]

                else:
                    ell = Ellipticity(thickness, velocity_p, velocity_s, density, self._algorithm, self._dc)
                    rel = ell(curve.period, mode=curve.mode)
                    dcalc = rel.ellipticity

                n = len(dcalc)
                error += curve.weight * numpy.sum((dcalc - curve.data[:n]) ** 2)

            except DispersionError:
                return numpy.Inf

        return error

    def _get_density(self, velocity_p):
        return numpy.array([self._density_func(vp) for vp in velocity_p])

    @property
    def layers(self):
        return self._layers

    @property
    def curves(self):
        return self._curves

    @property
    def n_layers(self):
        return len(self._layers)

    @property
    def n_curves(self):
        return len(self._curves)
