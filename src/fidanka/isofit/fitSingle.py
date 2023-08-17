from fidanka.bolometric.bctab import BolometricCorrector
from fidanka.isochrone.MIST import read_iso, read_iso_metadata
from fidanka.misc.logging import LoggerManager
from fidanka.misc.utils import interpolate_arrays

from scipy.spatial.distance import cdist
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
):
    logger = LoggerManager.get_logger()
    if rFilterOrder:
        f3Key = f1Key
    else:
        f3Key = f2Key

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
    bounds = [(np.inf, None), (np.inf, None)]
    ax = plt.gca()
    for age, isoAtAge in iso.items():
        color = isoAtAge[f1Key] - isoAtAge[f2Key]
        mag = isoAtAge[f3Key]
        minDist = np.min(
            cdist(
                np.array([starColor, starMag]).reshape(1, -1), np.array([color, mag]).T
            )
        )

        # get the direction of the minium distance point from starColor and starMag
        # if the point is above the isochrone, the distance is negative
        direction = np.sign(
            mag[
                np.argmin(
                    cdist(
                        np.array([starColor, starMag]).reshape(1, -1),
                        np.array([color, mag]).T,
                    )
                )
            ]
            - starMag
        )

        print(f"Age: {age}, Distance: {minDist}, Direction: {direction}")
        if direction > 0 and bounds[0][0] > minDist:
            bounds[1] = (minDist, age, (color, mag))
        elif direction < 0 and bounds[1][0] > minDist:
            bounds[0] = (minDist, age, (color, mag))

    ax.plot(bounds[0][2][0], bounds[0][2][1], ".", color="k", alpha=0.25)
    ax.plot(bounds[1][2][0], bounds[1][2][1], ".", color="k", alpha=0.25)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    LoggerManager.config_logger("Validate_singleStarFit.log")
    # testIsochrone = "../../../../localTests/testData/isos/MIST/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_HST_WFC3.iso.cmd"
    testIsochrone = "../../../../localTests/fidanka/validation/isos/PopA+0.24.txt"
    iso, meta = read_iso(testIsochrone), read_iso_metadata(testIsochrone)
    testStar = (0.46, 1.18)
    ax.plot(testStar[0], testStar[1], "o", color="r")
    FeH = 0.24

    import pathlib

    isos = pathlib.Path("../../../../localTests/testData/isos/MIST/").glob("*.iso.cmd")
    isoAtFeH = interpolate_iso_to_single_star_FeH(isos, FeH)

    estimate_single_star_age(
        testStar[0], testStar[1], isoAtFeH, FeH, "WFC3_UVIS_F606W", "WFC3_UVIS_F814W"
    )
    plt.show()
