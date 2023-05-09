#!/usr/bin/env python


from setuptools import setup, Extension
import numpy as np

ext = Extension(
    "fidanka.ext.nearest_neighbors",
    sources=["src/fidanka/ext/src/nearest_neighbors.c"],
    include_dirs=[np.get_include()],
)

if __name__ == "__main__":
    setup(ext_modules=[ext])
