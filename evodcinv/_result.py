import numpy

import matplotlib.pyplot as plt


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
