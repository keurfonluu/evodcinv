from .__about__ import __version__
from ._io import read
from ._model import EarthModel
from ._result import InversionResult

__all__ = [
    "EarthModel",
    "InversionResult",
    "read",
    "__version__",
]
