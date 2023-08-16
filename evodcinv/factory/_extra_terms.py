import numpy as np


def increasing_velocity(x, penalty=np.inf):
    """
    Strictly increasing velocity models.

    Parameters
    ----------
    x : array_like
        Model parameter vector.
    penalty : scalar, optional, default Inf
        Penalty applied to misfit of models with velocities that are not strictly increasing.

    Returns
    -------
    scalar
        Misfit value.

    """
    n_layers = (len(x) + 1) // 3
    vs = x[n_layers - 1 : 2 * n_layers - 1]

    return 0.0 if (vs[1:] >= vs[:-1]).all() else penalty


def prior(x, depth, velocity_s, uncertainties=None, alpha=1.0e-3):
    """
    Prior S-wave velocity model.

    Parameters
    ----------
    x : array_like
        Model parameter vector.
    depth : array_like
        Prior model's depths (in km).
    velocity_s : array_like.
        Prior model's S-wave velocities (in km/s).
    uncertainties : scalar, array_like or None, optional, default None
        Uncertainties associated to prior model S-wave velocities (in km/s).
    alpha : scalar, optional, default 1.0e-3
        Regularization factor.

    Returns
    -------
    scalar
        Misfit value.

    """
    n_layers = (len(x) + 1) // 3
    d = x[: n_layers - 1]
    z = np.insert(d.cumsum(), 0, 0.0)
    vs = x[n_layers - 1 : 2 * n_layers - 1]
    vprior = np.interp(z, depth, velocity_s)

    sigma = uncertainties if uncertainties is not None else 1.0
    sigma = np.interp(z, depth, uncertainties) if np.ndim(sigma) == 1 else sigma

    return alpha * np.square((vs - vprior) / sigma).sum()


def smooth(x, alpha=1.0e-3):
    """
    Smooth velocity models.

    Parameters
    ----------
    x : array_like
        Model parameter vector.
    alpha : scalar, optional, default 1.0e-3
        Regularization factor.

    Returns
    -------
    scalar
        Misfit value.

    """
    n_layers = (len(x) + 1) // 3
    vs = x[n_layers - 1 : 2 * n_layers - 1]

    return alpha * n_layers * np.square(vs[:-2] - 2.0 * vs[1:-1] + vs[2:]).sum()
