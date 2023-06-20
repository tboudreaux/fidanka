"""
=======
Fidanka
=======

Fiducual line generation along with isochrone fitting.

============
Installation
============
From Source
-----------

>>> git clone https://github.com/tboudreaux/fidanka.git
>>> cd fidanka
>>> pip install -e .

===============
Simple Examples
===============

Single Isochrone Fitting
------------------------
Imagine a file called HUGS1.csv which contains photometry and can be read with
pandas for the globular cluster NGC 2808 in your current working directory.
Also imagine a directory ISO which contains an isochron in the MIST format. We
can fit that isochrone to the photometry as follows

>>> from fidanka.isochrone.MIST import read_iso
>>> from fidanka.fiducual import fiducual_line
>>> from fidanka.isofit.fit import optimize
>>> import pandas as pd
>>> photometry = pd.read_csv("HUGS1.csv")
>>> iso = read_iso("ISO/mist.iso")
>>>
>>> filter1 = photometry["F275W"]
>>> filter2 = photometry["F814W"]
>>> error1 = photometry["F275W_RMS"]
>>> error2 = photometry["F814W_RMS"]
>>>
>>> fiducualLine = fiducual_line(filter1, filter2, error1, error2, reverseFilterOrder=True)
>>> bestFitResults = optimize(fiducualLine, iso, ("F275W", "F814W", "F814W"))
>>>
>>> print(bestFitResults)

"""
__version__ = "1.0.0"
