## 0.6.0a1 (2023-09-01)

### Feat

- **fitSingle**: added ability to fit the age of a single star
- **fit**: added error handling to the isochrone fitter and also improved results format and also black reformat
- **isofit**: added new objective functions and built in parellization and black reformatt
- **population**: cleaned up pop syth ageing by adding ager class and black reformated
- **bolometric,-population,-misc**: Fully updated pop synth to new method using artiificial star object and reworked sample function to be faster and black reformat
- **population**: added filter aliases to artificial star tests and began reworking population synthethis sampler to make it simpler and black reformated
- **population**: began major rework of population sythethis object to use artificial star objects, total cluster mass, and age object, and black reformat
- **artificialStar.py**: Finished adding position based completness estimate to artificialStar class and black reformatted
- **artificialStar.py**: filled out artificial star template, began work on completness estimation, black reformatted
- **artificialStar.py**: added error function building to artificialStar class and black reformatted
- **artificialStar.py**: began adding template for artificial star class to make that system more extensible and reformatted
- **fidanka.bolometric**: added checksum verification to MIST bolometric table fetch and reformat
- **load.py,-URLS.py**: added the ability to automatically fetch MIST bolometric correction tables + black reformated

### Fix

- **fit**: minor
- **fit**: added missing import
- **ager**: unknown
- **fiducual**: fixed bug and made width_coef interface compatible with older code and black reformat
- **isofit,-bolometric,-isofit,-misc**: brought isofit systems up to date with the new bolometric systems and fixed logging system filename to use singleton pattern, black reformatt
- **read_iso**: allow mist isochrones with bad ages to be read in and bad ages skipped and black reformat
- **utils.py**: fixed broken point function distance and reformated
- **python-build.py**: fixed coverage report to write out as json
- **python-build.yml**: added fidanka install step
- **python-build.yml**: fixed enviromental variable
- **synthesize.py**: fixed some undefined variable names + reformated
- **mc.py**: imported missing packages and fixed undefined variable names
- **fiducial.py**: imported missing package from c extensions

### Refactor

- **fit**: changed parallel result to be a tuple based record and not a key value pair
- **fiducial.py**: removed unused ridge_bounding function
- **runTest.sh**: renamed runTest to runTests
- **fidanka.modeling**: removed unused sub module

### Perf

- **bolometric**: added better cacheing, multithreading, and scope limiting to bolometric corrections and black reformat

## 0.6.0a0 (2023-06-23)

### Feat

- **utils.py**: Added interface for measuring the perpendicular distance between two functions over a domain

### Fix

- **setup.py,-pyproject.toml**: updated build system to use numpy and added numpy as a pre-dependency
- **models.py**: imported pfD from new file
- **utils.py**: added return value to function which should have had one
- **utils.py**: fixed tqdm keyword argument error
- **fiducial.py**: fixed vMagBinMeans definition location
- **fiducial.py**: Fixed bug where nPops = 1. Pulled from 200k33p3r/fidanka
- **pyproject.toml**: bumped minimum targeter version from 3.7 to 3.10 (allows pip insstall . to work with setuptools)

### Refactor

- removed unused functions
- **makefile**: pre-commit refactoring
- **test_utils.py**: reformated
- **test_utils.py**: black reformated
- black formatter
