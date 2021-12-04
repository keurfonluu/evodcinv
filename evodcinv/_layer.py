class Layer:
    def __init__(self, thickness, velocity_s, poisson=None):
        """
        Layer class.

        Parameters
        ----------
        thickness : array_like
            Layer thickness search boundary (in km).
        velocity_s : array_like
            Layer S-wave velocity search boundary (in km/s).
        poisson : array_like or None, optional, default None
            Layer Poisson's ratio search boundary.

        """
        poisson = poisson if poisson is not None else [0.2, 0.4]
        for arg in (thickness, velocity_s, poisson):
            assert len(arg) == 2
            assert arg[0] <= arg[1]

        self._thickness = tuple(thickness)
        self._velocity_s = tuple(velocity_s)
        self._poisson = tuple(poisson)

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
