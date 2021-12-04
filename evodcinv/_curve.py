import numpy
from disba._helpers import is_sorted


class Curve:
    def __init__(self, period, data, mode=0, wave="rayleigh", type="phase", weight=1.0, uncertainties=None):
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
        uncertainties : scalar, array_like or None, optional, default None
            Uncertainties associated to data points. If None, error will be normalized by `data`.

        """
        assert len(period) == len(data)
        assert mode >= 0
        assert wave in {"rayleigh", "love"}
        assert type in {"phase", "group", "ellipticity"}
        assert is_sorted(period)

        self._period = numpy.asarray(period)
        self._data = numpy.asarray(data)
        self._mode = mode
        self._wave = wave
        self._type = type
        self._weight = weight
        self._uncertainties = uncertainties

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
        if numpy.ndim(new_period) == 0:
            assert isinstance(new_period, int)

            new_period = numpy.linspace(self.period[0], self.period[-1], new_period)

        elif numpy.ndim(new_period) == 1:
            assert is_sorted(new_period)
            assert new_period[0] >= self.period[0]
            assert new_period[-1] <= self.period[-1]

        else:
            raise ValueError()

        new_data = numpy.interp(new_period, self.period, self.data)

        if inplace:
            self._data = new_data
            
        else:
            return Curve(new_period, new_data, self.mode, self.wave, self.type, self.weight, self.uncertainties)

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
