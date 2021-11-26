from .._helpers import register
from ._h5 import read, write

__all__ = [
    "read",
    "write",
]


register("h5", [".h5"], read, write)
