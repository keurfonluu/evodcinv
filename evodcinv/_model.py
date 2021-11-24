import numpy
from disba import DispersionError, Ellipticity, surf96
from disba._common import ifunc
from stochopy.optimize import minimize

from ._common import itype
from ._helpers import nafe_drake
from ._result import InversionResult
from ._progress import ProgressBar
from ._curve import Curve
from ._layer import Layer


class EarthModel:
    def __init__(self):
        self._layers = []
        self._configuration = {}

    def add(self, layer):
        assert isinstance(layer, Layer)

        self.layers.append(layer)

    def pop(self):
        assert self.n_layers

        return self.layers.pop(-1)

    def configure(self, optimizer="cpso", misfit="rmse", density="nafe-drake", dc=0.001, dt=0.01, optimizer_args=None, extra_terms=None):
        assert optimizer in {"cmaes", "cpso", "de", "na", "pso", "vdcma"}
        assert misfit in {"norm1", "norm2", "rmse"} or hasattr(misfit, "__call__")
        assert density in {"nafe-drake"} or hasattr(density, "__call__")
        assert dc > 0.0
        assert dt > 0.0

        optimizer_args = optimizer_args if optimizer_args is not None else {}
        assert isinstance(optimizer_args, dict)

        extra_terms = extra_terms if extra_terms is not None else []
        assert isinstance(extra_terms, (list, tuple))
        for extra_term in extra_terms:
            assert hasattr(extra_term, "__call__")

        # Misfit type
        if misfit == "norm1":
            misfit = lambda x: numpy.abs(x).sum()

        elif misfit == "norm2":
            misfit = lambda x: numpy.square(x).sum()

        elif misfit == "rmse":
            misfit = lambda x: (numpy.square(x).sum() / len(x)) ** 0.5

        # Density function
        if density == "nafe-drake":
            density = nafe_drake

        self._configuration = {
            "optimizer": optimizer,
            "misfit": misfit,
            "density": density,
            "dc": dc,
            "dt": dt,
            "optimizer_args": optimizer_args,
            "extra_terms": extra_terms,
        }

    def invert(self, curves, maxrun=1, split_results=False):
        assert self._configuration
        assert isinstance(curves, (list, tuple))
        for curve in curves:
            assert isinstance(curve, Curve)

        # Optimizer arguments
        method = self._configuration["optimizer"]
        optimizer_args = self._configuration["optimizer_args"]

        _optimizer_args = {
            "maxiter": 100,
            "popsize": 10,
            "seed": None,
        }
        _optimizer_args.update(optimizer_args)

        maxiter = _optimizer_args["maxiter"]
        popsize = _optimizer_args["popsize"]

        # Overwrite options
        constraints = {
            "cmaes": "Penalize",
            "cpso": "Shrink",
            "de": "Random",
            "pso": "Shrink",
            "vdcma": "Penalize",
        }

        _optimizer_args["return_all"] = True
        if method != "na":
            _optimizer_args["constraints"] = constraints[method]

        # Set random seed
        seed = _optimizer_args.pop("seed")
        if seed is not None:
            numpy.random.seed(seed)

        # Minimize misfit function
        func = lambda x: self._misfit_function(x, curves)
        bounds = numpy.vstack(
            [
                [layer.thickness for layer in self._layers[:-1]],
                [layer.velocity_s for layer in self._layers],
                [layer.poisson for layer in self._layers],
            ]
        )

        results = []
        for i in range(maxrun):
            prefix = f"Run {i + 1:<{len(str(maxiter)) - 1}d}"
            with ProgressBar(prefix, max=maxiter) as bar:
                def callback(X, res):
                    bar.misfit = res.fun
                    bar.next()

                x = minimize(func, bounds, method=method, options=_optimizer_args, callback=callback)

            # Parse output
            velocity_models = numpy.empty((maxiter, popsize, self.n_layers, 4))
            for i in range(x.nit):
                for j in range(popsize):
                    velocity_models[i, j] = numpy.transpose(
                        self.transform(x.xall[i, j])
                    )

            result = InversionResult(
                xs=numpy.concatenate(x.xall),
                models=numpy.concatenate(velocity_models),
                misfits=numpy.concatenate(x.funall),
                global_misfits=numpy.minimum.accumulate(x.funall.min(axis=1)),
                maxiter=maxiter,
                popsize=popsize,
            )
            results.append(result)

        if split_results:
            return results

        else:
            out = InversionResult()
            for result in results:
                out += result

            return out

    def transform(self, x):
        thickness = x[: self.n_layers - 1]
        velocity_s = x[self.n_layers - 1 : 2 * self.n_layers - 1]
        poisson = x[2 * self.n_layers - 1 :]
        velocity_p = self._get_velocity_p(velocity_s, poisson)
        density = self._get_density(velocity_p)

        return numpy.append(thickness, 1.0), velocity_p, velocity_s, density

    def _misfit_function(self, x, curves):
        thickness, velocity_p, velocity_s, density = self.transform(x)
        misfit = self._configuration["misfit"]
        dc = self._configuration["dc"]
        dt = self._configuration["dt"]
        extra_terms = self._configuration["extra_terms"]

        error_extra = 0.0
        if len(extra_terms):
            for extra_term in extra_terms:
                error_extra += extra_term(x)

            if numpy.isinf(error_extra):
                return numpy.Inf

        error = 0.0
        weights_sum = 0.0
        for curve in curves:
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
                        ifunc["dunkin"][curve.wave],
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
                        dc=dc,
                    )
                    rel = ell(curve.period, mode=curve.mode)
                    dcalc = numpy.abs(rel.ellipticity)

                n = len(dcalc)
                sigma = (
                    curve.data[:n]
                    if curve.uncertainties is None
                    else curve.uncertainties[:n]
                )
                error += curve.weight * misfit((dcalc - curve.data[:n]) / sigma)
                weights_sum += curve.weight

            except DispersionError:
                return numpy.Inf

        return error / weights_sum + error_extra

    def _get_density(self, velocity_p):
        try:
            return numpy.array([self._configuration["density"](vp) for vp in velocity_p])

        except KeyError:
            raise RuntimeError("model is not configured")

    @staticmethod
    def _get_velocity_p(velocity_s, poisson):
        return velocity_s * ((1.0 - poisson) / (0.5 - poisson)) ** 0.5

    @property
    def layers(self):
        return self._layers

    @property
    def n_layers(self):
        return len(self._layers)
