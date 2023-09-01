from scipy.optimize import minimize
from fidanka.bolometric.bctab import BolometricCorrector
from fidanka.isochrone.MIST import read_iso, read_iso_metadata
from fidanka.misc.logging import LoggerManager
from fidanka.misc.utils import interpolate_arrays
from fidanka.isochrone.isochrone import interp_isochrone_age
from fidanka.isofit.fit import shortest_distance_from_point_to_function

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from collections import deque

import numpy as np
import pandas as pd


def identify_FeH_bounding_paths(isochroneSet, FeH):
    logger = LoggerManager.get_logger()

    logger.info("Interpolating Isochrone to Single Star FeH")
    # get the two closest isochrones to the star's FeH
    FeHAbove, FeHBelow = np.inf, -np.inf
    for isoPath in isochroneSet:
        meta = read_iso_metadata(isoPath)
        if meta["[Fe/H]"] == FeH:
            logger.info("Exact Isochrone Found")
            return (isoPath, meta), (isoPath, meta)
        elif meta["[Fe/H]"] > FeH and meta["[Fe/H]"] < FeHAbove:
            logger.info(f'Updating FeHAbove to {meta["[Fe/H]"]}')
            isoAbove, metaAbove = (isoPath, meta)
            FeHAbove = meta["[Fe/H]"]
        elif meta["[Fe/H]"] < FeH and meta["[Fe/H]"] > FeHBelow:
            logger.info(f'Updating FeHBelow to {meta["[Fe/H]"]}')
            isoBelow, metaBelow = (isoPath, meta)
            FeHBelow = meta["[Fe/H]"]
    return (isoBelow, metaBelow), (isoAbove, metaAbove)


def interpolate_iso_to_single_star_FeH(isochroneSet, FeH):
    logger = LoggerManager.get_logger()

    lower, upper = identify_FeH_bounding_paths(isochroneSet, FeH)

    logger.info(
        f"Lower Isochrone [Fe/H]: {lower[1]['[Fe/H]']}, Upper Isochrone: {upper[1]['[Fe/H]']}"
    )
    lowerIsoSet, lowerIsoMeta = read_iso(lower[0]), lower[1]
    upperIsoSet, upperIsoMeta = read_iso(upper[0]), upper[1]

    newIso = dict()
    for (lowerAge, lowerIso), (upperAge, upperIso) in zip(
        lowerIsoSet.items(), upperIsoSet.items()
    ):
        assert lowerAge == upperAge, "Isochrone Age Mismatch!"

        isoAtFeH = interpolate_arrays(
            lowerIso.values,
            upperIso.values,
            FeH,
            lowerIsoMeta["[Fe/H]"],
            upperIsoMeta["[Fe/H]"],
            joinCol=0,
        )
        newIso[lowerAge] = isoAtFeH

    for age, iso in newIso.items():
        newIso[age] = pd.DataFrame(iso, columns=lowerIso.columns)

    return newIso


def get_point_iso_dist(starColor, starMag, isoAtAge, fKey1, fKey2, rFilterOrder=False):
    isoColor = isoAtAge[fKey1] - isoAtAge[fKey2]
    if rFilterOrder:
        isoMag = isoAtAge[fKey2]
    else:
        isoMag = isoAtAge[fKey1]
    isoF = interp1d(isoMag, isoColor, bounds_error=False, fill_value="extrapolate")
    dist = shortest_distance_from_point_to_function(starMag, starColor, isoF)
    return dist


def get_dist_between_iso_and_star_at_age(
    iso, age, starColor, starMag, fKey1, fKey2, rFilterOrder=False
):
    isoHeader = iso[list(iso.keys())[0]].columns
    isoAtAge = interp_isochrone_age(iso, age)
    isoAtAge = pd.DataFrame(isoAtAge, columns=isoHeader)
    dist = get_point_iso_dist(
        starColor, starMag, isoAtAge, fKey1, fKey2, rFilterOrder=rFilterOrder
    )
    return dist[0]


def get_init_age_guess(iso, starColor, starMag, fKey1, fKey2, rFilterOrder=False):
    guesses = list()
    for age, isoAtAge in iso.items():
        dist = get_point_iso_dist(
            starColor, starMag, isoAtAge, fKey1, fKey2, rFilterOrder=rFilterOrder
        )
        guesses.append((age / 1e9, dist))
    guesses = sorted(guesses, key=lambda x: x[1])
    return guesses[0][0]


def estimate_single_star_age(
    starColor,
    starMag,
    isochrones,
    FeH,
    f1Key,
    f2Key,
    rFilterOrder=False,
    bcFilterSystem=None,
    mu=0,
    Av=0,
    Rv=3.1,
    ageBounds=[5, 15],
):
    """
    Estimate the age of a single star based on its color, magnitude, and isochrones.

    Parameters
    ----------
    starColor : float
        The color index of the star.
    starMag : float
        The apparent magnitude of the star.
    isochrones : dict
        A dictionary containing age-related theoretical isochrones.
    FeH : float
        Metallicity [Fe/H] of the star.
    f1Key : str
        The key for filter 1 in the isochrone data.
    f2Key : str
        The key for filter 2 in the isochrone data.
    rFilterOrder : bool, default=False
        Whether to reverse the filter order.
    bcFilterSystem : str, default=None
        The filter system to use for bolometric corrections.
    mu : float, default=0
        The distance modulus.
    Av : float, default=0
        The extinction in magnitudes.
    Rv : float, default=3.1
        The ratio of total to selective extinction.
    ageBounds : list, default=[5, 15]
        The lower and upper age bounds for the estimation.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The results of the optimization to find the estimated age.

    Notes
    -----
    - If `bcFilterSystem` is provided, bolometric corrections are applied to the isochrones.
    - The function uses the `minimize` function from `scipy.optimize` to optimize the age estimate.
    """
    logger = LoggerManager.get_logger()

    if bcFilterSystem is not None:
        logger.info("Applying Bolometric Correction to Isochrone")
        iso = dict()
        bc = BolometricCorrector(bcFilterSystem, FeH)
        for age, theoreticalIso in isochrones.items():
            correctedIso = bc.apparent_mags(
                10 ** theoreticalIso["log_Teff"],
                theoreticalIso["log_g"],
                theoreticalIso["log_L"],
                mu=mu,
                Av=Av,
                Rv=Rv,
            )
            iso[age] = correctedIso
    else:
        logger.info("No Bolometric Correction Applied")
        iso = isochrones
    # bound the age of the point between two isochrones before optimizing it
    initAgeGuess = get_init_age_guess(
        iso, starColor, starMag, f1Key, f2Key, rFilterOrder=rFilterOrder
    )
    minResults = minimize(
        lambda age: get_dist_between_iso_and_star_at_age(
            iso,
            age,
            starColor,
            starMag,
            fKey1=f1Key,
            fKey2=f2Key,
            rFilterOrder=rFilterOrder,
        ),
        initAgeGuess,
        bounds=[ageBounds],
    )
    return minResults


if __name__ == "__main__":
    LoggerManager.config_logger("Validate_singleStarFit.log")
    # testIsochrone = "../../../../localTests/testData/isos/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_HST_WFC3.iso.cmd"
    testIsochrone = "../../../../localTests/fidanka/validation/isos/PopA+0.24.txt"
    iso, meta = read_iso(testIsochrone), read_iso_metadata(testIsochrone)
    testStar = (0.46, 1.18)
    FeH = 0.24

    import pathlib

    isos = pathlib.Path("../../../../localTests/testData/isos/MIST/").glob("*.iso.cmd")
    isoAtFeH = interpolate_iso_to_single_star_FeH(isos, FeH)

    age = estimate_single_star_age(
        testStar[0], testStar[1], isoAtFeH, FeH, "WFC3_UVIS_F606W", "WFC3_UVIS_F814W"
    )
    print(age)
