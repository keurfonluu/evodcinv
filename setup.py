# -*- coding: utf-8 -*-

from setuptools import find_packages
from numpy.distutils.core import setup, Extension

VERSION = "1.0.0"
DISTNAME = "evodcinv"
DESCRIPTION = "EvoDCinv"
LONG_DESCRIPTION = """Dispersion curve inversion using Evolutionary Algorithms"""
AUTHOR = "Keurfon LUU"
AUTHOR_EMAIL = "keurfon.luu@mines-paristech.fr"
URL = "https://github.com/keurfonluu/evodcinv"
LICENSE = "MIT License"
REQUIREMENTS = [
    "numpy",
    "matplotlib",
    "stochopy>=1.7.0",
]
CLASSIFIERS = [
    "Programming Language :: Python",
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
]

FFLAGS = "-O3 -ffast-math -funroll-loops -fno-protect-parens -fopenmp"

ext1 = Extension(
    name = "evodcinv._dispcurve",
    sources = ["evodcinv/f90/dispcurve.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    extra_link_args = [ "-lgomp" ],
    )

ext2 = Extension(
    name = "evodcinv._lay2vel",
    sources = ["evodcinv/f90/lay2vel.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    f2py_options = [],
    )

if __name__ == "__main__":
    setup(
        name = DISTNAME,
        description = DESCRIPTION,
        long_description = LONG_DESCRIPTION,
        author = AUTHOR,
        author_email = AUTHOR_EMAIL,
        url = URL,
        license = LICENSE,
        install_requires = REQUIREMENTS,
        classifiers = CLASSIFIERS,
        version = VERSION,
        packages = find_packages(),
        include_package_data = True,
        ext_modules = [ ext1, ext2 ],
    )
