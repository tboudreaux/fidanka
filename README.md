<p align="center">
	<img width="460" height="460" src="/assets/fidankaLogo.png">
</p>

---

<p align="center">

[![Python - Linux](https://github.com/tboudreaux/fidanka/actions/workflows/python-build.yml/badge.svg)](https://github.com/tboudreaux/fidanka/actions/workflows/python-build.yml)

[![Forks](https://img.shields.io/github/forks/tboudreaux/fidanka.svg?style=for-the-badge)](https://github.com/tboudreaux/fidanka)
[![Stars](https://img.shields.io/github/stars/tboudreaux/fidanka.svg?style=for-the-badge)](https://github.com/tboudreaux/fidanka)

</p>

---

# Fidanka
Fidanka is an (hopefully) easy to use python package for doing science with globular clusters.
Written by Emily M. Boudreaux and Martin Ying.


## Docs & Information
The full API Documentation can be found <a href="https://tboudreaux.github.io/fidanka/">here</a>.

A short presentation which I've given on fidanka may be found <a href="https://algebrist.ddns.net/~tboudreaux/presentations/VAST/#/">here</a>

## Install

### PyPi
Fidnka will be avalible on PyPi soon; however, it is not currently avalible anywhere other than GitHub.

### From Source (latest)

```bash
git clone https://github.com/tboudreaux/fidanka.git
cd fidanka
pip install .
```

### Development
In order to develop for fidanka it will be helpful to also install a few more packages. We also recommend you work in a virtual environment.
Moreover, you must work in your own fork of fidanka and issue pull requests. First fork fidanka on GitHub.
```bash
conda create --name fidankaDev python=3
pip install pre-commit
pip install commitizen
git clone https://github.com/<your-username>/fidanka.git
pip install -e .
pre-commit install
pre-commit autoupdate
```
While use commitizen is not required when contributing, it is highly encouraged.
If you do use commitizen then the add commit workflow is as follows

```bash
git add <files>
cz c
```

## Contributing
We welcome any contributions made to this project. Please work in a fork and submit a pull request. We also request that you use the pre-commit hooks we have defined and commit using commitizen. This will help us keep a cleaner channelog. Finally, any contributors to this project are expected to behave in accordance with the code of conduct (code_of_conduct.md). Any violation of this will result in a ban on contributions and an evaluation of whether previous contributions should be purged.

## Examples

### Measuring the fiducial lines of a cluster with multiple populations
Assuming you have your photometry stored in some data structure (here I retrieve it from a pickle
as a pandas dataframe), and you have a prior that there are 2 populations within the
cluster, you can measure those fiducial lines as follows

This measurement will re-sample the data 1000 times and remeasure the fiducial lines each time
in order to get confidence intervals and a mean.
```python
from fidanka.fiducial import measure_fiducial_lines
import pickle as pkl

PHOTROOT = "photometry.pkl"
with open(PHOTROOT, 'rb') as f:
    photometry = pkl.load(f)[1]

MC = 1000
fiducialLine = measure_fiducial_lines(
        photometry["F275W"],
        photometry["F814W"],
        photometry["F275W_RMS"],
        photometry["F814W_RMS"],
        reverseFilterOrder=True,
        mcruns=MC,
        nPops = 2
        )
popA = fiducualLine[0]
popB = fiducualLine[1]

popAMean = popA.mean
popA5th = popA.confidence(0.05)
popA95th = popA.confidence(0.95)

with open('fldump-sorted.pkl', 'wb') as f:
    pkl.dump(fiducialLine, f)
```

### Fitting an Isochrone

Imagine a file called HUGS1.csv which contains photometry and can be read with
pandas for the globular cluster NGC 2808 in your current working directory.
Also imagine an isochrone called iso.txt (In the MIST format) in the current working directory. We
can fit that isochrone to the photometry as follows. Also imagine we still
have popA loaded from the previous example. Finally, imagine you have a series of bolometric
correction tables in the current directory stored in a folder called "bolTables".
These tables should be in the format available on the MIST website.

```python
from fidanka.isochrone.MIST import read_iso
from fidanka.isofit.fit import optimize

import pandas as pd
import re
import os

bolFilenames = list(filter(lambda x: re.search("feh[mp]\d+", x), os.listdir("bolTables")))
bolPaths = list(map(lambda x: os.path.join(args.bTabs, x), bolFilenames))
FeH = ...

photometry = pd.read_csv("HUGS1.csv")
iso = read_iso("ISO/mist.iso")

filter1 = photometry["F275W"]
filter2 = photometry["F814W"]
error1 = photometry["F275W_RMS"]
error2 = photometry["F814W_RMS"]

bestFitResults = optimize(
    popA.mean,
    iso,
    bolPaths,
    FeH,
    filters = ("F275W", "F814W", "F814W")
)

print(bestFitResults)
```

### logging
fidanka will write some information to stdout; however, more extensive information will be writte
to a log file. By default this is called fidanka.log; however, its name and log level can be
configured with the get_logger function

```python
from fidanka.misc.utils import get_logger
import logging

get_logger("rootLoggerName", "testRun.log", clevel=logging.INFO)
```

This will result in much more information being written to std out. The first
argument is the name of the logger module and can be whatever you like. The second
is the filename for the file handler. There are also keyword arguments
clevel, flevel, and level which control the minimum logger level to be written
to the console, the file, and either respectively.


### Population Synthethis
Assume you have a series of isochrones loaded in a list and you want to generate
a 12.5Gyr cluster with 30000 members and a binary fraction of 0.25.

```python
from fidanka.population.synthesize import population
from fidanka.fiducial.fiducial import measure_fiducial_line


ARTSTARTEST = "ArtificialStarCalibratedErrorFunctions.pkl"
with open("./RMSFuncs.pkl", 'rb') as f:
    rmsFuncs = pkl.load(f)
with open(ARTSTARTEST, 'rb') as f:
    artStar = pkl.load(f)


targetAge = 12.5e9
n = 30000

pop = population(
    isos[:2],
    -0.84,
    0.25,
    lambda x: x-x + targetAge,
    n,
    targetAge,
    targetAge,
    0.25,
    2,
    artStar,
    9198,
    0.17,
    "F606W"
)

# Note that the population Synthesis runs when the code
# gets here, NOT at population instantiation time
# this can also be called with pop.data().
# If this function is called multiple times the same results
# will be returned for the same object as a cache is used
df = pop.to_pandas()
pop.to_csv("TestPop.csv")
```
