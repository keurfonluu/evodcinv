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
                "misfit".rjust(m) + ": " + repr(self["misfit"]),
                "x".rjust(m) + ": " + repr(self["x"]),
                "model".rjust(m) + ": " + repr(self["model"]),
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
            misfit = min(self.misfit, other.misfit)
            if misfit == self.misfit:
                x = self.x
                model = self.model

            else:
                x = other.x
                model = other.model

            return InversionResult(
                misfit=misfit,
                x=x,
                model=model,
                all_x=numpy.vstack([self.all_x, other.all_x]),
                all_models=numpy.vstack([self.all_models, other.all_models]),
                all_misfits=numpy.concatenate([self.all_misfits, other.all_misfits]),
                global_misfits=numpy.row_stack((self.global_misfits, other.global_misfits)),
                maxiter=([self.maxiter] if isinstance(self.maxiter, int) else self.maxiter) + ([other.maxiter] if isinstance(other.maxiter, int) else other.maxiter),
                popsize=([self.popsize] if isinstance(self.popsize, int) else self.popsize) + ([other.popsize] if isinstance(other.popsize, int) else other.popsize),
            )

    def write(self, filename, indent=None):
        from ._io import write

        write(filename, self, indent)
