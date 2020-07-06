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
    x_axis: str. "frequency" or "period".
        Choose between a frequency of a period as x-axis for the dispersion
        curve.
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


###TODO: inutile je crois...
def _frequency_conversion(func):
    """ 
    Wrapper function. 

    Wraps conversion from period to frequency if needed.
    """
    def _conversion_needed(x_axis):
        if x_axis == "period":
            return func
        elif:
            def _convert(*args):
                args[0] = np.sort(1./ args)
                return func(*args)
            return _convert(*args)
        else:
            raise ValueError("x_axis must be either 'period' or 'Frequency'.")

