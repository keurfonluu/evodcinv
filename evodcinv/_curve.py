import numpy as np
from disba._helpers import is_sorted


class Curve:
    def __init__(
        self,
        period,
        data,
        mode=0,
        wave="rayleigh",
        type="phase",
        weight=1.0,
        uncertainties=1.0,
    ):
        """
        Curve data class.

        Parameters
        ----------
        period : array_like
            Periods (in s).
        data : array_like
            Data array to fit.
        mode : int, optional, default 0
            Mode number (0 if fundamental).
        wave : str {'love', 'rayleigh'}, optional, default 'rayleigh'
            Wave type.
        type : str {'phase', 'group', 'ellipticity'}, optional, default 'phase'
            Data type.
        weight : scalar, optional, default 1.0
            Overall weight applied to the misfit error associated to this data set.
        uncertainties : scalar or array_like, optional, default 1.0
            Uncertainties associated to data points.

        """
        if len(period) != len(data):
            raise ValueError()
        if mode < 0:
            raise ValueError()
        if wave not in {"rayleigh", "love"}:
            raise ValueError()
        if type not in {"phase", "group", "ellipticity"}:
            raise ValueError()
        if not is_sorted(period):
            raise ValueError()

        self._period = np.asarray(period)
        self._data = np.asarray(data)
        self._mode = mode
        self._wave = wave
        self._type = type
        self._weight = weight
        self._uncertainties = uncertainties if uncertainties is not None else 1.0

    def __repr__(self):
        """Pretty curve."""
        out = []
        out += [f"{80 * '-'}"]
        out += [f"{self.wave.capitalize()}-wave {self.type}"]
        out[-1] += " velocity\n" if self.type != "ellipticity" else "\n"
        out += [f"Number of data points: {self.data.size}"]
        out += [f"Period bounds: [{self.period.min():.4f}, {self.period.max():.4f}] s"]
        out += [f"Mode: {self.mode}"]
        out += [f"{80 * '-'}"]

        return "\n".join(out)

    def resample(self, new_period, inplace=False):
        """
        Resample curve.

        Parameters
        ----------
        new_period : scalar or array_like
            Periods at which to generate new data.
        inplace : bool, optional, default True
            If `False`, return a new :class:`evodcinv.Curve`.

        Returns
        -------
        evodcinv.Curve
            New data curve (only if ``inplace == False``).

        """
        if np.ndim(new_period) == 0:
            if not isinstance(new_period, int):
                raise TypeError()

            new_period = np.linspace(self.period[0], self.period[-1], new_period)

        elif np.ndim(new_period) == 1:
            if not is_sorted(new_period):
                raise ValueError()
            if new_period[0] < self.period[0]:
                raise ValueError()
            if new_period[-1] > self.period[-1]:
                raise ValueError()

        else:
            raise ValueError()

        new_data = np.interp(new_period, self.period, self.data)

        if inplace:
            self._data = new_data

        else:
            return Curve(
                new_period,
                new_data,
                self.mode,
                self.wave,
                self.type,
                self.weight,
                self.uncertainties,
            )

    @property
    def period(self):
        """Return periods."""
        return self._period

    @property
    def data(self):
        """Return data array."""
        return self._data

    @property
    def mode(self):
        """Return mode."""
        return self._mode

    @property
    def wave(self):
        """Return wave type."""
        return self._wave

    @property
    def type(self):
        """Return data type."""
        return self._type

    @property
    def weight(self):
        """Return misfit error weight."""
        return self._weight

    @property
    def uncertainties(self):
        """Return uncertainties associated to data points."""
        return self._uncertainties
