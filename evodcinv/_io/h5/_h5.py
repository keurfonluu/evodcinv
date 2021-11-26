import h5py

import numpy

from ..._result import InversionResult


def read(filename):
    with h5py.File(filename, "r") as f:
        result = {k: v[0] if numpy.ndim(v) == 0 else numpy.array(v) for k, v in f.items()}

    return InversionResult(**result)


def write(filename, result, compression="gzip", compression_opts=4):
    with h5py.File(filename, "w") as f:
        for k, v in result.items():
            if numpy.ndim(v) == 0:
                f.create_dataset(k, data=(v,), compression=compression, compression_opts=compression_opts)

            else:
                f.create_dataset(k, data=v, compression=compression, compression_opts=compression_opts)
