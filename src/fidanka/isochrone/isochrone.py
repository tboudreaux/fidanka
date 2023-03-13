import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Tuple, Union

import pickle as pkl
import pathlib
import re

from tqdm import tqdm

from scipy.interpolate import interp1d

FARRAY_1D = npt.NDArray[np.float64]

def shift_isochrone(
        color : Union[FARRAY_1D, pd.Series],
        magnitude : Union[FARRAY_1D, pd.Series],
        distance : float,
        extinction : float
        ) -> Tuple[Union[FARRAY_1D, pd.Series], Union[FARRAY_1D, pd.Series]]:
    """
    Shift the isochrone around in distance and reddining. Note! This is currently
    wrong. I have not had time to fix it yet but be aware that to get true
    distances and reddinings out this function will need to be fixed!

    TODO: FIX

    Parameters
    ----------
        color : Union[ndarray[float64], pd.Series]
            1D array of color of shape m
        magniude : Union[ndarray[float64], pd.Series]
            1D array of magnitude of shape m
        distance : float
            Distance, in parsecs, to shift the isochrone by
        extinction : float
            Color excess to shift the isochrone by. NOTE THAT THIS IS
            NOT CURRENTLY IMPLIMENTED CORRECTLY. WILL NOT AFFECT TESTING
            WHICH IS WHY I HAVE NOT FIXED IT; HOWEVER, WILL PREVETNT TRUE
            DISTANCES OR REDDENINGS FROM BEING EXTRACTED

    Returns
    -------
        aptMag : Union[ndarray[float64], pd.Series]
            Apparent magnitude, shifted by the distance
        aptCol : Union[ndarray[float64], pd.Series]
            Apparent color, shifted by distance and reddining
    """
    mu = 5*np.log10(distance) - 5 + extinction
    aptMag = mu + magnitude
    aptCol = 3.2*extinction + color
    return aptMag, aptCol

def load_ISO_CMDs(
        root : str
        ) -> dict[str, dict[float, dict[float, pd.DataFrame]]]:
    """
    Load pysep formated Isochrones into a lookup table indexed by population
    name helium mass fraction, and alpha enhancement.

    Parameters
    ----------
        root : str
            Root to search for CMD.pkl files from

    Returns
    -------
        lookup : dict[str, dict[float, dict[float, pd.DataFrame]]]
            lookup table for pysep formated isochrones indexed by population
            name, helium mass fraction, and alpha enhancement.
    """
    CMDs = list(map(lambda x: str(x), pathlib.Path(root).rglob("CMD.pkl")))
    extract = list(map(lambda x: re.findall(r"Pop(A|E)\+(\d\.\d+)\/alpha-(\d\.\d+)\/", x)[0], CMDs))
    pops = set(map(lambda x: x[0], extract))
    Ys = set(map(lambda x: x[1], extract))
    alphas = set(map(lambda x: x[2], extract))

    lookup = dict()
    for pop in tqdm(pops, leave=False):
        lookup[pop] = dict()
        for Y in tqdm(Ys, leave=False):
            lookup[pop][float(Y)] = dict()
            for alpha in tqdm(alphas, leave=False):
                if (pop, Y, alpha) in extract:
                    extractID = extract.index((pop, Y, alpha))
                    with open(CMDs[extractID], 'rb') as f:
                        CMD = pkl.load(f)
                    lookup[pop][float(Y)][float(alpha)] = CMD
    return lookup

def interCMDatMag(
        color : Union[FARRAY_1D, pd.Series],
        mag : Union[FARRAY_1D, pd.Series],
        targetMag : float
        ) -> Union[float, FARRAY_1D]:
    """
    Given some color and magnitude, assumed to be relativley clean and from an
    isochrone shifted into the approproate filters. Then linearlly interpolate,
    parmeterized by magnitude, to the target magnitude.

    Parameters
    ----------
        color : Union[ndarray[float64], pd.Series]
            1D array of color of shape m
        mag : Union[ndarray[float64], pd.Series]
            1D array of magnitude of shape m
        targetMag : float
            Magnitude to evaluate interpolation result at.

    Returns
    -------
        colorAtMag : Union[float, ndarray[float64]]
            colors of the isochrone at the requested magnitude(s)

    """
    f = interp1d(mag, color)
    colorAtMag = f(targetMag)
    return colorAtMag

def interp_isochrone_age(
        iso : dict[float, pd.DataFrame],
        targetAge : float
        ) -> pd.DataFrame:
    """
    Given some dictionary of isochrones where the keys are the ages of those
    isochrones in Gyr, interpolate between isochrones of two bounding ages to
    find the isochrones at a target age. Do this using pandas efficient apply
    functions. Works for MIST and pysep formated isochrones so long as they
    have been loaded into pandas using the routines in this library.

    Parameters
    ----------
        iso : dict[float, pd.DataFrame]
            Dictionary of isochrones indexed by isochrones age in Gyr
        targetAge : float
            Target age to interpolate to. This age must be greater than the
            minimum age and less than the max age in iso.

    Returns
    -------
        interpolate : pd.DataFrame
            Isochrones with all columns and rows interpolated to the targetAge
            linearlly from the upper and lower bounding aged isochrones
    """
    logTargetAgeYr= np.log10(targetAge*1e9)
    ageKeys = list(iso.keys())
    distance = [(x-logTargetAgeYr, x) for x in ageKeys]
    below = sorted(filter(lambda x: x[0] <=0, distance), key=lambda x: abs(x[0]))
    above = sorted(filter(lambda x: x[0] > 0, distance), key=lambda x: x[0])
    isoBelow = iso[below[0][1]]
    isoAbove = iso[above[0][1]]

    assert isoBelow is not None
    assert isoAbove is not None

    assert 'log10_isochrone_age_yr' in isoAbove
    assert 'log10_isochrone_age_yr' in isoBelow

    age1 = isoBelow['log10_isochrone_age_yr'].iloc[0]
    age2 = isoAbove['log10_isochrone_age_yr'].iloc[0]

    def linearinterpolate(x, other, age1, age2):
        newIso = ((other[x.name] - x)/(age2-age1)) * (logTargetAgeYr - age1) + x
        return newIso

    interpolated = isoBelow.apply(lambda x: linearinterpolate(x, isoAbove, age1, age2))

    assert isinstance(interpolated, pd.DataFrame)
    return interpolated
