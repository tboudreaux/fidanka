# Fidanka

Routines for generating fiducual lines and fitting isochrones to those fiducual
lines

## API Docs
https://algebrist.com/~tboudreaux/docs/fidanka/fidanka.fiducial.html

## Install

### From Source

```bash
git clone https://github.com/tboudreaux/fidanka.git
cd fidanka
pip install -e .
```

## Example

Imagine a file called HUGS1.csv which contains photometry and can be read with
pandas for the globular cluster NGC 2808 in your current working directory.
Also imagine a directory ISO which contains an isochron in the MIST format. We
can fit that isochrone to the photometry as follows

```python
from fidanka.isochrone.MIST import read_iso
from fidanka.fiducual import fiducual_line
from fidanka.isofit.fit import optimize
import pandas as pd

photometry = pd.read_csv("HUGS1.csv")
iso = read_iso("ISO/mist.iso")

filter1 = photometry["F275W"]
filter2 = photometry["F814W"]
error1 = photometry["F275W_RMS"]
error2 = photometry["F814W_RMS"]

fiducualLine = fiducual_line(filter1, filter2, error1, error2, reverseFilterOrder=True)
bestFitResults = optimize(fiducualLine, iso, ("F275W", "F814W", "F814W"))

print(bestFitResults)
```


# Notes
Currently my implimentation of shifting the isochrone around due to galactic
reddening is done wrong. This should be a really easy fix I just haven't had
time to get to it since I can debug other stuff without fixing that. Don't
trust distnces or reddening's you get until fixing that (I've just mixed up Av,
E(B-V) and the filters I'm using...should be an easy fix). I'll get to it in
the next week or two and/or you are welcome to make the change and copmmit it
to the repo.

Also, I've extensivley documented the code. In the next few days I'll use
sphinx to build the documentation to a web page so there is an easy reference
to use. 
