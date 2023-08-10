import numpy as np


class Layer:
    def __init__(self, thickness, velocity_s, poisson=None):
        """
        Layer class.

        Parameters
        ----------
        thickness : scalar or array_like
            Layer thickness search boundary (in km).
        velocity_s : scalar or array_like
            Layer S-wave velocity search boundary (in km/s).
        poisson : scalar, array_like or None, optional, default None
            Layer Poisson's ratio search boundary.

        """
        poisson = poisson if poisson is not None else [0.2, 0.4]
        kwargs = {
            "thickness": thickness,
            "velocity_s": velocity_s,
            "poisson": poisson,
        }

        for k, v in kwargs.items():
            ndim = np.ndim(v)

            if ndim == 0:
                if not isinstance(v, (int, float)):
                    raise ValueError()

                v = (v, v)

            elif ndim == 1:
                if len(v) != 2:
                    raise ValueError()

                if v[0] > v[1]:
                    raise ValueError()

            setattr(self, f"_{k}", tuple(v))

    def __repr__(self):
        """Pretty layer."""
        out = []

        # Table header
        out += [f"{80 * '-'}"]
        out += ["Layer parameters\n"]

        out += [f"{60 * '-'}"]
        out += [f"{'d [km]'.rjust(20)}{'vs [km/s]'.rjust(20)}{'nu [-]'.rjust(20)}"]
        out += [3 * f"{'min'.rjust(10)}{'max'.rjust(10)}"]

        # Table
        out += [f"{60 * '-'}"]

        d_min, d_max = self.thickness
        vs_min, vs_max = self.velocity_s
        nu_min, nu_max = self.poisson
        out += [
            f"{d_min:>10.4f}{d_max:>10.4f}{vs_min:>10.4f}{vs_max:>10.4f}{nu_min:>10.4f}{nu_max:>10.4f}"
        ]

        out += [f"{60 * '-'}"]

        out += [f"{80 * '-'}"]

        return "\n".join(out)

    @property
    def thickness(self):
        """Return layer thickness search boundary."""
        return self._thickness

    @property
    def velocity_s(self):
        """Return layer S-wave velocity search boundary."""
        return self._velocity_s

    @property
    def poisson(self):
        """Return layer Poisson's ratio search boundary."""
        return self._poisson
