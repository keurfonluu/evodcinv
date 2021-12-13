import numpy as np

nafe_drake_poly = np.poly1d([0.000106, -0.0043, 0.0671, -0.4721, 1.6612, 0.0])


def nafe_drake(velocity_p):
    """Nafe-Drake's correlation for density."""
    return nafe_drake_poly(velocity_p)
