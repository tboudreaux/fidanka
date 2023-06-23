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
