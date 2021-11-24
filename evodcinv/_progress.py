import numpy

from progress.bar import IncrementalBar


class ProgressBar(IncrementalBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.width = 20
        self.suffix = "%(percent)3d%% [%(elapsed_td)s / %(eta_td)s] - Misfit: %(misfit).4f"
        self.check_tty = False
        self.misfit = numpy.Inf

    @property
    def misfit(self):
        return self._misfit

    @misfit.setter
    def misfit(self, value):
        self._misfit = value
