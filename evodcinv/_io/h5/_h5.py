import h5py
import numpy as np

from ..._result import InversionResult


def read(filename):
    """
    Import H5 file with inversion results.

    Parameters
    ----------
    filename : str
        Input file name.

    Returns
    -------
    :class:`evodcinv.InversionResult`
        Inversion results.

    """
    with h5py.File(filename, "r") as f:
        result = {k: v[0] if np.ndim(v) == 0 else np.array(v) for k, v in f.items()}

    return InversionResult(**result)


def write(filename, result, compression_opts=4):
    """
    Export inversion results to H5.

    Parameters
    ----------
    filename : str
        Output file name.
    result : :class:`evodcinv.InversionResult`
        Inversion results to export.
    compression_opts : int, optional, default 4
        Compression level for gzip compression. May be an integer from 0 to 9.

    """
    with h5py.File(filename, "w") as f:
        for k, v in result.items():
            if np.ndim(v) == 0:
                f.create_dataset(k, data=(v,), compression="gzip", compression_opts=compression_opts)

            else:
                f.create_dataset(k, data=v, compression="gzip", compression_opts=compression_opts)
