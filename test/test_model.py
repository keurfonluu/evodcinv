import numpy as np
from helpers import curves, model


def test_invert():
    res = model.invert(curves)

    assert np.allclose(res.model.sum(), 8.90561595)
