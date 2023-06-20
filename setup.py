#!/usr/bin/env python
from setuptools import setup, Extension


class get_numpy_include(object):
    def __str__(self):
        import numpy

        return numpy.get_include()


ext = Extension(
    "fidanka.ext.nearest_neighbors",
    sources=["src/fidanka/ext/src/nearest_neighbors.c"],
    include_dirs=[get_numpy_include()],
)

if __name__ == "__main__":
    setup(name="fidanka", ext_modules=[ext])
