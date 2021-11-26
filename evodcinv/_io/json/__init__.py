from .._helpers import register
from ._json import read, write

__all__ = [
    "read",
    "write",
]


register("json", [".json"], read, write)
