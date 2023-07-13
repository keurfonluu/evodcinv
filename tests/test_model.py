import copy

import helpers
import numpy as np
import pytest


@pytest.mark.parametrize(
    "xref, increasing_velocity", [(8.90561595, False), (12.76397397, True)]
)
def test_invert(xref, increasing_velocity):
    model = copy.deepcopy(helpers.model)
    model.configure(
        optimizer="cpso",
        misfit="rmse",
        density="nafe-drake",
        optimizer_args={
            "popsize": 5,
            "maxiter": 5,
            "seed": 0,
        },
        increasing_velocity=increasing_velocity,
    )
    res = model.invert(helpers.curves)
    assert np.allclose(res.model.sum(), xref)

    # Check threshold
    rest = res.threshold(10.0)

    assert np.allclose(res.x, rest.x)
    assert np.allclose(res.model, rest.model)
    assert np.allclose(res.misfit, rest.misfit)
