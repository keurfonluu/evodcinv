import pathlib

__all__ = [
    "__version__",
]


with open(f"{pathlib.Path(__file__).parent}/VERSION") as f:
    __version__ = f.readline().strip()
