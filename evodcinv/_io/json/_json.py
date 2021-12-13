import json

import numpy as np

from ..._result import InversionResult


def read(filename):
    """
    Import json file with inversion results.

    Parameters
    ----------
    filename : str
        Input file name.

    Returns
    -------
    :class:`evodcinv.InversionResult`
        Inversion results.

    """
    with open(filename, "r") as f:
        result = json.load(f)

    result = {k: v if np.ndim(v) == 0 else np.array(v) for k, v in result.items()}

    return InversionResult(**result)


def write(filename, result, indent=None):
    """
    Export inversion results to json.

    Parameters
    ----------
    filename : str
        Output file name.
    result : :class:`evodcinv.InversionResult`
        Inversion results to export.
    indent : int, str or None, optional, default None
        Indent level.

    """
    from copy import deepcopy

    def jsonify(x):
        """JSON serialize data."""
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        elif isinstance(x, (list, tuple)):
            return [jsonify(xx) for xx in x]
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, dict):
            return {k: jsonify(v) for k, v in x.items()}
        else:
            return x

    with open(filename, "w") as f:
        result = deepcopy(result)
        result = jsonify(result)
        json.dump(result, f, indent=indent)
