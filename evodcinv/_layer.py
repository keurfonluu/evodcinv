class Layer:
    def __init__(self, thickness, velocity_s, poisson=[0.2, 0.4]):
        for arg in (thickness, velocity_s, poisson):
            assert len(arg) == 2
            assert arg[0] <= arg[1]

        self._thickness = tuple(thickness)
        self._velocity_s = tuple(velocity_s)
        self._poisson = tuple(poisson)

    @property
    def thickness(self):
        return self._thickness

    @property
    def velocity_s(self):
        return self._velocity_s

    @property
    def poisson(self):
        return self._poisson
