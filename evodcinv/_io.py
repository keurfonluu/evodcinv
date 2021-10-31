import json

import numpy

from ._helpers import InversionSummary


def read(filename):
    with open(filename, "r") as f:
        summary = json.load(f)

    summary = {
        k: v if numpy.ndim(v) == 0 else numpy.array(v)
        for k, v in summary.items()
    }

    return InversionSummary(**summary)


def write(filename, summary, indent=None):
    from copy import deepcopy

    def jsonify(x):
        """JSON serialize data."""
        if isinstance(x, (numpy.int32, numpy.int64)):
            return int(x)
        elif isinstance(x, (list, tuple)):
            return [jsonify(xx) for xx in x]
        elif isinstance(x, numpy.ndarray):
            return x.tolist()
        elif isinstance(x, dict):
            return {k: jsonify(v) for k, v in x.items()}
        else:
            return x

    with open(filename, "w") as f:
        summary = deepcopy(summary)
        summary = jsonify(summary)
        json.dump(summary, f, indent=indent)
