from disba import GroupDispersion, PhaseDispersion

from functools import wraps, partial

__all__ = [ "Dispersion" ]

def Dispersion(
            dtype="phase",
            algorithm="dunkin",
            dc=0.005,
            dt=0.025,
):
    """
    Wrapper for the Disba classes.

    Parameters
    ----------
    dtype: str. "phase" or "group".
        Choose between group or phase velocity dispersion curves.
    algorithm : str {'dunkin', 'fast-delta'}, optional, default 'dunkin'
        Algorithm to use for computation of Rayleigh-wave dispersion:
         - 'dunkin': Dunkin's matrix (adapted from surf96),
         - 'fast-delta': fast delta matrix (after Buchen and Ben-Hador, 1996).
    dc : scalar, optional, default 0.005
        Phase velocity increment for root finding.
    dt : scalar, optional, default 0.025. Useless if dtype="phase".
        Frequency increment (%) for calculating group velocity.

    """

    if dtype == "phase":
        partial_constructor = lambda a,b,c,d: PhaseDispersion(a, b, c, d,
                algorithm, dc) 

    elif dtype == "group":
        partial_constructor = lambda a,b,c,d: GroupDispersion(a, b, c, d,
                algorithm, dc, dt)
    else:
        raise ValueError("This kind of dispersion curve is not specified")

    return partial_constructor

