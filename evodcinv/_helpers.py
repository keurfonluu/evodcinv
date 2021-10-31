import numpy

from ._io import write


nafe_drake_poly = numpy.poly1d([0.000106, -0.0043, 0.0671, -0.4721, 1.6612, 0.0])


class InversionSummary(dict):
    def write(self, filename, indent=None):
        write(filename, self, indent)

    def __getattr__(self, name):
        """Define dict.attr as an alias of dict[attr]."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Pretty result."""
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1

            return "\n".join([
                "misfit".rjust(m) + ": " + repr(self["misfit"]),
                "x".rjust(m) + ": " + repr(self["x"]),
                "model".rjust(m) + ": " + repr(self["model"]),
            ])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        """Return a list of attributes."""
        return list(self.keys())


def get_velocity_p(velocity_s, poisson):
    return velocity_s * ((1.0 - poisson) / (0.5 - poisson)) ** 0.5


def nafe_drake(velocity_p):
    return nafe_drake_poly(velocity_p)
