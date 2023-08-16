import numpy as np
from progress.bar import IncrementalBar


class ProgressBar(IncrementalBar):
    def __init__(self, *args, **kwargs):
        """Custom progress bar with misfit information."""
        super().__init__(*args, **kwargs)

        self.width = 20
        self.suffix = (
            "%(percent)3d%% [%(elapsed_td)s / %(eta_td)s] - Misfit: %(misfit)s"
        )
        self.check_tty = False
        self.misfit = np.Inf

    @property
    def misfit(self):
        """Return misfit value."""
        return self._misfit

    @misfit.setter
    def misfit(self, value):
        self._misfit = value
