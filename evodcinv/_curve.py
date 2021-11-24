import numpy
from disba._helpers import is_sorted


class Curve:
    def __init__(self, period, data, mode=0, wave="rayleigh", type="phase", weight=1.0, uncertainties=None):
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
        return self._period

    @property
    def data(self):
        return self._data

    @property
    def mode(self):
        return self._mode

    @property
    def wave(self):
        return self._wave

    @property
    def type(self):
        return self._type

    @property
    def weight(self):
        return self._weight

    @property
    def uncertainties(self):
        return self._uncertainties
