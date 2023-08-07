from fidanka.misc.utils import interpolate_keyed_arrays
from fidanka.misc.logging import LoggerManager
from fidanka.isochrone.MIST import read_iso, read_iso_metadata
import numpy as np
import numpy.typing as npt
import pandas as pd

from typing import Tuple, Union

import pickle as pkl
import pathlib
import re

from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.integrate import quad

FARRAY_1D = npt.NDArray[np.float64]


def shift_full_iso(
    iso: np.ndarray,
    distance: float,
    reddening: float,
    effectiveWavelengths: list,
    columnNames: list,
    responseFunctions: dict,
):
    """
    Shift the isochrone around in distance and reddining.
    """
    mu = 5 * np.log10(distance) - 5
    shifted = np.zeros_like(iso)
    for ID, (filterValues, eW, name) in enumerate(
        zip(iso.T, effectiveWavelengths, columnNames)
    ):
        # print(filterValues)
        responseFunction = responseFunctions[name]
        print(f"Integrating from {responseFunction[1]} to {responseFunction[2]}")
        df = lambda l: responseFunction[0](l)
        da = lambda l: (reddening * 3.1) * calc_extinction_coef(l)
        F = quad(df, responseFunction[1], responseFunction[2])[0]
        A = quad(da, responseFunction[1], responseFunction[2])[0]
        norm = 1 / F
        print(f"Normalization factor: {norm}")
        A = norm * A
        # import matplotlib.pyplot as plt
        # plt.plot(np.linspace(responseFunction[1], responseFunction[2], 1000), responseFunction[0](np.linspace(responseFunction[1], responseFunction[2], 1000)))
        # plt.title(f"Filter {ID} ({name})")
        # plt.xlabel("Wavelength (nm)")
        # plt.ylabel("Response")
        # plt.show()
        print(f"Total response in filter {ID} ({name}): {F}")
        print(f"Total extinction in filter {ID} ({name}): {A}")
        dM = filterValues + A + mu
        print(f"Total extinction in filter {ID} ({name}): {dM}")
        print(f"Shifted filter {ID}({name}) by {mu} + {dM} = {mu+dM}")
        shifted[:, ID] = filterValues + mu + dM
        exit()

        # print(eW)
        # X = calc_extinction_coef(eW)
        # print(f"Extinction coefficient for filter {ID} @ {eW}nm: {X}")
        # A = (X*3.1)*reddening
        # print(f"Extinction in filter {ID}: {A}")
        # shifted[:, ID] = filterValues + mu + A
        # print(f"Shifted filter {ID} by {mu} + {A} = {mu+A}")
        # print(f"Shifted filter {ID} by {mu+A} = {shifted[:, ID]}")
    return shifted


def shift_isochrone(
    color: Union[FARRAY_1D, pd.Series],
    magnitude: Union[FARRAY_1D, pd.Series],
    distance: float,
    reddening: float,
    filter1EffectiveWavelength: float = 275,
    filter2EffectiveWavelength: float = 814,
    rFilterOrder: bool = True,
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
        reddining : float
            Reddining, in magnitudes E(B-V)
        filter1EffectiveWavelength : float, default=275
            Effective wavelength of the first filter in nanometers
        filter2EffectiveWavelength : float, default=814
            Effective wavelength of the second filter in nanometers
        rFilterOrder : bool, default=True
            If true reverse the standard filter order. This is used for
            isochrones that are in the HST filter order.


    Returns
    -------
        aptMag : Union[ndarray[float64], pd.Series]
            Apparent magnitude, shifted by the distance
        aptCol : Union[ndarray[float64], pd.Series]
            Apparent color, shifted by distance and reddining
    """
    mu = 5 * np.log10(distance) - 5
    print(f"Distance: {distance} pc")
    print(f"Distance modulus: {mu}")
    X1 = calc_extinction_coef(filter1EffectiveWavelength)
    X2 = calc_extinction_coef(filter2EffectiveWavelength)
    print(f"Extinction coefficient for filter 1: {X1}")
    print(f"Extinction coefficient for filter 2: {X2}")
    A1 = X1 * reddening
    A2 = X2 * reddening
    print(f"Color excess: {reddening}")
    print(f"Extinction in filter 1: {A1}")
    print(f"Extinction in filter 2: {A2}")
    aptCol = color + (A1 - A2)
    print(f"Apparent color: {aptCol}")
    if rFilterOrder:
        aptMag = magnitude + mu + A2
    else:
        aptMag = magnitude + mu + A1
    return aptMag, aptCol


def calc_extinction_coef(wEffective, Rv=3.1):
    """
    Calculate the extinction coefficient for a given wavelength in nanometers.
    This is used to calculate the color excess of an isochrone.

    Calculations are based on the CCM 1989 extinction law.
    Cardelli, Clayton, and Mathis 1989, ApJ, 345, 245

    Parameters
    ----------
        wEffective : float
            Effective wavelength in nanometers
        Rv : float
            Ratio of total to selective extinction. Default is 3.1

    Returns
    -------
        extinctionCoef : float
            Extinction coefficient for the given wavelength
    """
    print(f"Effective wavelength: {wEffective} nm")
    w = wEffective / 1000
    print(f"Effective wavelength: {w} um")
    iL = 1 / (w)
    print(f"iL: {iL}")
    if 0.3 <= iL <= 1.1:
        print("Coefficients for infrared")
        a = 0.574 * iL**1.61
        b = -0.527 * iL**1.61
    elif 1.1 <= iL <= 3.3:
        print("Coefficients for optical")
        y = iL - 1.82
        a = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )
        print(f"y: {y}")
        print(f"a: {a}")
        print(f"b: {b}")
    elif 3.3 <= iL <= 8:
        print("Coefficients for ultraviolet")
        if 5.9 <= iL <= 8:
            Fa = -0.04473 * (iL - 5.9) ** 2 - 0.009779 * (iL - 5.9) ** 3
            Fb = 0.2130 * (iL - 5.9) ** 2 + 0.1207 * (iL - 5.9) ** 3
        elif iL < 5.9:
            Fa = Fb = 0
        else:
            raise ValueError("Wavelength out of range")
        a = 1.752 - 0.316 * iL - 0.104 / ((iL - 4.67) ** 2 + 0.341) + Fa
        b = -3.090 + 1.825 * iL + 1.206 / ((iL - 4.62) ** 2 + 0.263) + Fb
    else:
        raise ValueError("Wavelength out of range")
    return a + (b / Rv)


def load_ISO_CMDs(root):
    """
    Load the isochrone color-magnitude diagrams (CMDs) and metadata from a specified root directory.

    This function recursively searches the root directory for isochrone text files ("isochrones.txt").
    It organizes the data by population type (PopA or PopE), helium content (Y), and alpha enhancement (alpha).
    It returns two nested dictionaries. The first contains the CMDs and the second contains the metadata
    corresponding to the CMDs.

    Parameters
    ----------
    root : str
        The root directory to begin the search for "isochrones.txt" files.

    Returns
    -------
    lookup : dict
        A nested dictionary with the CMD data, indexed by the population type, helium content, and alpha
        enhancement. The structure is {population: {helium: {alpha: CMD}}}.
    FeHLookup : dict
        A nested dictionary with the CMD metadata, indexed by the population type, helium content, and alpha
        enhancement. The structure is {population: {helium: {alpha: metadata}}}.

    Notes
    -----
    The function uses regular expressions to extract the population type, helium content, and alpha enhancement
    from the file path. It assumes the path is structured as follows: "Pop(A|E)+(Y)/alpha-(alpha)".

    Progress bars from the tqdm module are displayed during the operation to provide visual feedback.

    The 'read_iso' and 'read_iso_metadata' functions are assumed to be available in the same scope.

    If a tuple of parameters (population, helium, alpha) is not found in the list of extracted parameters,
    it is skipped and not included in the dictionaries.

    The function uses the walrus operator ':=' which is only available in Python 3.8 and later.
    """
    logger = LoggerManager.get_logger()

    logger.info(f"Identifying isochrone CMDs from {root}. This may take some time.")
    CMDs = list(map(lambda x: str(x), pathlib.Path(root).rglob("isochrones.txt")))
    logger.info(f"Found {len(CMDs)} CMDs")
    extract = list(
        map(lambda x: re.findall(r"Pop(A|E)\+(\d\.\d+)\/alpha-(\d\.\d+)\/", x)[0], CMDs)
    )
    pops = set(map(lambda x: x[0], extract))
    Ys = set(map(lambda x: x[1], extract))
    alphas = set(map(lambda x: x[2], extract))

    logger.info("Organizing CMDs by population, Y, and alpha")
    logger.info(f"Populations: {pops}")
    logger.info(f"Helium contents: {Ys}")
    logger.info(f"Alpha enhancements: {alphas}")
    logger.info("This may take a few minutes...")

    lookup = dict()
    FeHLookup = dict()
    for pop in tqdm(pops, leave=False):
        lookup[pop] = dict()
        FeHLookup[pop] = dict()
        for Y in tqdm(Ys, leave=False):
            lookup[pop][float(Y)] = dict()
            FeHLookup[pop][float(Y)] = dict()
            for alpha in tqdm(alphas, leave=False):
                if checkTup := (pop, Y, alpha) in extract:
                    logger.info(f"Loading CMD for {pop}+{Y}/alpha-{alpha}")
                    extractID = extract.index((pop, Y, alpha))
                    iso = read_iso(CMDs[extractID])
                    metadata = read_iso_metadata(CMDs[extractID])
                    FeH = metadata["[Fe/H]"]
                    lookup[pop][float(Y)][float(alpha)] = iso
                    FeHLookup[pop][float(Y)][float(alpha)] = FeH
                    logger.info(f"Loaded CMD for {pop}+{Y}/alpha-{alpha}")
    return lookup, FeHLookup


def interCMDatMag(
    color: Union[FARRAY_1D, pd.Series],
    mag: Union[FARRAY_1D, pd.Series],
    targetMag: float,
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
    iso: dict[float, pd.DataFrame], targetAgeGyr: float
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
        targetAgeGyr : float
            Target age to interpolate to (in Gyr). This age must be greater than the
            minimum age and less than the max age in iso.

    Returns
    -------
        interpolate : pd.DataFrame
            Isochrones with all columns and rows interpolated to the targetAge
            linearlly from the upper and lower bounding aged isochrones
    """
    targetAgeYr = targetAgeGyr * 1e9
    ageKeys = list(iso.keys())
    distance = [(x - targetAgeYr, x) for x in ageKeys]
    below = sorted(filter(lambda x: x[0] <= 0, distance), key=lambda x: abs(x[0]))
    above = sorted(filter(lambda x: x[0] > 0, distance), key=lambda x: x[0])

    age1 = below[0][1]
    age2 = above[0][1]
    isoBelow = iso[age1]
    isoAbove = iso[age2]

    age1 = np.log10(age1)
    age2 = np.log10(age2)
    logTargetAgeYr = np.log10(targetAgeYr)

    if isinstance(isoBelow, pd.DataFrame):
        isoBelow = isoBelow.values
    if isinstance(isoAbove, pd.DataFrame):
        isoAbove = isoAbove.values
    interpolated = interpolate_keyed_arrays(
        isoBelow, isoAbove, logTargetAgeYr, age1, age2, key=0
    )

    return interpolated


def iso_color_mag(iso, filter1, filter2, reverseFilterOrder: bool = False):
    if reverseFilterOrder:
        filter3 = filter2
    else:
        filter3 = filter1
    isoFilter1Name = f"WFC3_UVIS_{filter1}_MAG"
    isoFilter2Name = f"WFC3_UVIS_{filter2}_MAG"
    isoFilter3Name = f"WFC3_UVIS_{filter3}_MAG"

    isoColor = iso[isoFilter1Name] - iso[isoFilter2Name]
    isoMag = iso[isoFilter3Name]

    return isoColor, isoMag
