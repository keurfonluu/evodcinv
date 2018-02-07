# -*- coding: utf-8 -*-

from setuptools import find_packages
from numpy.distutils.core import setup, Extension

VERSION = "0.0.1"
DISTNAME = "evoseispy"
DESCRIPTION = "EvoSeisPy"
LONG_DESCRIPTION = """Seismology with Evolutionary Algorithm in Python"""
AUTHOR = "Keurfon LUU"
AUTHOR_EMAIL = "keurfon.luu@mines-paristech.fr"
URL = "https://github.com/keurfonluu/evoseispy"
LICENSE = "MIT License"
REQUIREMENTS = [
    "numpy",
    "matplotlib",
    "fteikpy>=1.2.3",
    "stochopy>=1.6.0",
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
    name = "evoseispy.dispersion_curve._dispcurve",
    sources = ["evoseispy/dispersion_curve/f90/dispcurve.f90"],
    extra_f90_compile_args = FFLAGS.split(),
    extra_link_args = [ "-lgomp" ],
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
        ext_modules = [ ext1 ],
    )
