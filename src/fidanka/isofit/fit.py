import pickle as pkl
from typing import Any, Callable, List, Tuple, Union, Dict

from fastdtw import fastdtw
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import dual_annealing, minimize
from scipy.optimize import OptimizeResult, minimize_scalar
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from fidanka.bolometric.bctab import BolometricCorrector
from fidanka.fiducial import fiducial_line
from fidanka.fiducial.fiducial import fiducial_line
from fidanka.isochrone.MIST import read_iso, read_iso_metadata
from fidanka.isochrone.isochrone import shift_isochrone
from fidanka.isochrone.isochrone import interp_isochrone_age
from fidanka.misc.utils import get_logger
from fidanka.misc.logging import LoggerManager

from concurrent.futures import ProcessPoolExecutor, as_completed
from queue import Queue

FARRAY_1D = npt.NDArray[np.float64]
R2_VECTOR = npt.NDArray[[np.float64, np.float64]]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]

CHI2R = dict[str, Any]
ORT = Tuple[float, Tuple[float, float, float], str, float, float]

BOUNDSDICT = {0: [2, 25], 1: [0.0001, 2], 2: [1, 30]}


def guess_mu(
    fIso: Callable,
    fFid: Callable,
    isoMagRange: Tuple[float, float],
    fidMagRange: Tuple[float, float],
    nMin: int = 5,
    nMax: int = 20,
    mMin: float = -10,
    mMax: float = 10,
    pbar: bool = False,
) -> Tuple[float, float]:
    """
    Use Dynamic Time Warping to estimate the mu adjustment between the fiducial
    line and isochrone optimization. This algorithm is somewhat convoluted; however,
    it works well. The steps are as follows
        1. Generate a grid of ns. These represent the number of points which will
           be used to run a latter interpolation step. The reason for this approach
           as compared to just using one larger n is that the scaling of dtw is O(n^2).
           However, by averaging the subsequent results over many small ns,
           we can get a good estimate.
        2. At each n we run something similar to a cross-correlation. However, instead
           of the peasron correlation we calculate the dtw distance and path between
           the two functions.
           2a. Using the path, which will connect equivilent features such as the MSTO
               from the isochrone to the fiducial line, we can calculate the average
               euclidian shift between equivilent points on the two functions.
        3. We take magnitude shift which results in the best (lowest) average euclidian
           disttance between the two functions.
        4. Over all ns we average these best magitude shits.

    I've tested this algorithm on a large number of cases, some of which are quite gnarlly.
    And it seems to work well. It is also quite fast. The only downside is that it is
    the initial location of the curves must be somewhat close. This can be easily
    achived using mean magnitude alignment.

    Parameters
    ----------
        fIso : Callable
            The isochrone function. This function must take a magnitude and return a color.
        fFid : Callable
            The fiducial line function. This function must take a magnitude and return a color.
        isoMagRange : Tuple[float, float]
            The magnitude range of the isochrone.
        fidMagRange : Tuple[float, float]
            The magnitude range of the fiducial line.
        nMin : int, default=5
            The minimum number of points to use in the interpolation.
        nMax : int, default=20
            The maximum number of points to use in the interpolation.
        mMin : float, default=-10
            The minimum magnitude shift to consider.
        mMax : float, default=10
            The maximum magnitude shift to consider.
        pbar : bool, default=False
            Whether or not to show a progress bar.

    Returns
    -------
        mean : float
            The mean magnitude shift.
        std : float
            The standard deviation of the magnitude shift.
    """
    logger = get_logger("fidanka.isofit.fit.guess_mu")
    logger.info("DTW mu adjustment estimation")

    Ns = list(range(nMin, nMax))
    bestMagShifts = np.zeros(shape=(len(Ns)))

    for nID, n in tqdm(
        enumerate(Ns), total=len(Ns), desc="Guessing mu", disable=not pbar
    ):
        correlations = list()

        magShifts = np.linspace(mMin, mMax, n)
        fMagEval = np.linspace(fidMagRange[0], fidMagRange[1], n)
        iMagEval = np.linspace(isoMagRange[0], isoMagRange[1], n)
        dtwX = np.zeros(shape=(n, 1))
        dtwY = np.zeros(shape=(n, 1))

        minFiducialColor, maxFiducialColor = min(fMagEval), max(fMagEval)
        fColor = fFid(fMagEval)
        dtwX[:, 0] = fColor

        # Cross DTW Correlate
        for magShift in magShifts:
            iColor = fIso(iMagEval + magShift)
            dtwY[:, 0] = iColor
            dtw = fastdtw(dtwX, dtwY, dist=euclidean)

            dists = list()
            for i, j in dtw[1]:
                if minFiducialColor <= iMagEval[j] <= maxFiducialColor:
                    dists.append(
                        np.sqrt(
                            (fColor[i] - iColor[j]) ** 2
                            + (fMagEval[i] - iMagEval[j]) ** 2
                        )
                    )

            chi2 = np.sum([x**2 for x in dists])
            chi2nu = chi2 / len(dists)
            correlations.append(chi2nu)

        # Select the mag result in the best (smallest) chi2nu
        magShift = magShifts[np.argmin(correlations)]
        bestMagShifts[nID] = magShift

    mean, std = np.mean(bestMagShifts), np.std(bestMagShifts)
    logger.info(f"DTW mu adjustment estimation {mean:0.3f} +/- {std:0.4f}")
    return mean, std


def get_init_mu_guess(
    iso: npt.NDArray,
    fiducialLine: npt.NDArray,
    bc: BolometricCorrector,
    filters: Tuple[str, str, str],
) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Make a first pass guess at the distance modulus using mean alignment
    of the magnitude at Av=0. Further optimization will have to be done
    but this preprocessing allows other algorithms such as the
    cross dtw algorithm to work

    Parameters
    ----------
        iso : np.ndarray
            The isochrones (MIST format). This must have the temperature
            as the 2nd column, the logg as the third, and the log L as
            the first (the first column should be the EEPs; however, those
            are not actually used in this function).
        fiducualLine : np.ndarray
            The fiducial line to be fit to. This should be a 2D array where
            the first column is the color of each fiducial point and the
            second column is the magnitude of each fiducial point. Therefore
            the shape will be (nx2) where n is the number of fiducial points.
        bc : BolometricCorrector
            A bolometric corrector object which will be used to shift the isochrone
            into apparent magnitude space.
        filters : Tuple[str, str, str]
            A tuple of string representing the two magnitudes used to make color
            and the magnitude for the vertical axis. The reason there are three
            and not two is to allow for easily swapping the order of which
            color component is the magnitude (as for example is common in the
            F275-F814W CMD). The color will be filter[0]-filter[1] and the
            mag will be filter[3].

    Returns
    -------
        shift : float
            The guessed distance modulus based on the mean magnitude alignment
            of the fiducial line and the bolometrically corrected isochrone
        isoMag : np.ndarray
            The bolometrically corrected magnitude shifted by the distance
            modulus guess.
        isoColor : np.ndarray
            The bolometrically corrected color.
    """
    logger = get_logger("fidanka.isofit.fit.get_init_mu_guess")
    logger.info("Initial mu guess")

    bolCorrected = bc.apparent_mags(iso[:, 1], iso[:, 2], iso[:, 3], Av=0, mu=0)

    isoColor = bolCorrected[filters[0]] - bolCorrected[filters[1]]
    isoMag = bolCorrected[filters[2]]
    shift = np.mean(isoMag) - np.mean(fiducialLine[1])

    logger.info(f"Initial mu guess {shift:0.3f}")

    return shift, (isoMag - shift, isoColor)


def limit_mu_space(
    iso: npt.NDArray,
    fiducialLine: npt.NDArray,
    bc: BolometricCorrector,
    filters: Tuple[str, str, str],
    fFid: Callable,
    pbar: bool = False,
) -> Tuple[float, float]:
    """
    Use the two step, mean mangitude alignment and dtw cross correlation to guess
    the distance modulus which should be applied to some isochchrone. Also guess
    the scatter in this distance modulus. This can be used to limit the search
    space for latter true optimization algorithms.

    Parameters
    ----------
        iso : np.ndarray
            The isochrones (MIST format). This must have the temperature
            as the 2nd column, the logg as the third, and the log L as
            the first (the first column should be the EEPs; however, those
            are not actually used in this function).
        fiducualLine : np.ndarray
            The fiducial line to be fit to. This should be a 2D array where
            the first column is the color of each fiducial point and the
            second column is the magnitude of each fiducial point. Therefore
            the shape will be (nx2) where n is the number of fiducial points.
        bc : BolometricCorrector
            A bolometric corrector object which will be used to shift the isochrone
            into apparent magnitude space.
        filters : Tuple[str, str, str]
            A tuple of string representing the two magnitudes used to make color
            and the magnitude for the vertical axis. The reason there are three
            and not two is to allow for easily swapping the order of which
            color component is the magnitude (as for example is common in the
            F275-F814W CMD). The color will be filter[0]-filter[1] and the
            mag will be filter[3].
        fFid : Callable
            The fiducial line function. This function must take a magnitude and return a color.
        pbar : bool, default=False
            Whether or not to show a progress bar.

    Returns
    -------
        totalShift : float
            The distance modulus (not guarrenteed to have the correct sign, you may
            have to take the absoulte value) which best minimized the dtw cross
            correlation function on average
        magStd : float
            The one sigma standard deviation in the distance modulus optimization
    """
    logger = get_logger("fidanka.isofit.fit.limit_mu_space")
    logger.info("Limiting mu search space")

    initMuGuess, (isoMag, isoColor) = get_init_mu_guess(iso, fiducialLine, bc, filters)

    f = interp1d(isoMag, isoColor, bounds_error=False, fill_value="extrapolate")

    isoMagRange = (min(isoMag), max(isoMag))
    fiducialMagRange = (min(fiducialLine[1]), max(fiducialLine[1]))

    magShiftCorrection, magStd = guess_mu(
        f, fFid, isoMagRange, fiducialMagRange, pbar=pbar
    )
    totalShift = initMuGuess + magShiftCorrection

    logger.info(f"Limiting mu search space {totalShift:0.3f} +/- {magStd:0.4f}")

    return totalShift, magStd


def bol_correct_iso(
    iso: npt.NDArray,
    bc: BolometricCorrector,
    filters: Tuple[str, str, str] = ("F606W", "F814W", "F606W"),
    Av: float = 0,
    distance: float = 0,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Bolometrically correct an isochrone and return the color and magnitude
    arrays. The isochrone must be in the correct format (i.e. the MIST format).

    Parameters
    ----------
        iso : np.ndarray
            The isochrone to bolometrically corerect, must be a numpy representation
            of a MIST format isochrone with EEPs in the 0th column, effective
            temperature (NOT log Teff) in the 1st, logg in the 2nd, and logL in the
            3rd.
        bc : BolometricCorrector
            An already instantiaed bolometric corrector object.
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            The filters to use for the color and magnitude. The first two
            are the color filters and the last is the magnitude filter.
        Av : float, default=0
            The color excess.
        distance : float, default=0
            The distance modulus.

    Returns
    -------
        isoColor : np.ndarray
            The bolometrically corrected color.
        isoMag : np.ndarray
            The bolometrically corrected magnitude.
    """
    logger = get_logger("fidanka.isofit.fit.bol_correct_iso")

    bolCorrected = bc.apparent_mags(iso[:, 1], iso[:, 2], iso[:, 3], Av=Av, mu=distance)

    isoColor = bolCorrected[filters[0]] - bolCorrected[filters[1]]
    isoMag = bolCorrected[filters[2]]
    return isoColor, isoMag


def get_ISO_CMD_Chi2(
    iso: pd.DataFrame,
    fiducialLine: FARRAY_2D_2C,
    bc: BolometricCorrector,
    fFid: Callable,
    filters: Tuple[str, str, str] = ("F606W", "F814W", "F606W"),
    distance: float = 0,
    Av: float = 0,
    age: float = None,
    n: int = 100,
    verbose: bool = False,
) -> float:
    """
    Calculate the reduced chi2 value between an isochrone and a fiducialLine
    at a given distance in parsecs and color excess. The Chi2 is calculated
    as the sum of the squares of the distance between equivilent evolutionary
    points as detected by dynamic time warping.

    Parameters
    ----------
        iso : pd.DataFrame
            The isochrone. If loaded and bolometrically corrected using pysep
            then this will be in the correct format. Otherwise make sure that
            it includes filters called "WFC3_UVIS_filter[0..2]_MAG" which have
            your filters in them. In future this will be cleaned up to allow for
            more general filters.
        fiducialLine : np.ndarray[[np.ndarray[float64], np.ndarray[float64]]]
            Fiducual line in the format output by the fiducual function. Where
            fiducialLine[:, 0] are the colors of each fiducual point and
            fiducialLine[:, 1] are the magniudes of each fiducual point.
        fFid : Callable
            The fiducial line function. This function must take a magnitude and return a color.
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            Filter names to use (which will get injcected to form the column
            names used to get the color and mag. The color is defined as
            filter[0] - filter[1] and the mag is filter[2])
        distance : float, default = 0
            Distance in parsecs to shift isochrone by when calculating the chi2
            value
        Av : float, default = 0
            Color excess to shift isochrone by when calculating the chi2 value
        age : float, default = None
            The Age Currently being optimized. Not used here directly; however, if passed
            will be used as an additional debug output.
        n : int, default = 100
            The number of points to use when calculating the interpolation
            used for the final dtw distance match (which is used to calculate
            the final chi2).
        verbose : bool, default = False
            Flag controlling whether verbose output will be used. This can
            be helpful for debugging; however, it may slow down the code.

    Returns
    -------
        chi2nu : float
            Reduced chi2 value. Calculated from the sum of the squares of the
            distances between EEPs between the fiducual line and
            isochrone as identified by dtw. Chi2 reduction is preformed then by dividing the chi2
            value by the number of points used to calculate it.
    """
    logger = get_logger("fidanka.isofit.fit.get_ISO_CMD_Chi2")
    isoColor, isoMag = bol_correct_iso(
        iso, bc, filters=filters, Av=Av, distance=distance
    )

    transposedFiducial = fiducialLine.T

    sortedFiducial = transposedFiducial[transposedFiducial[:, 1].argsort()]

    maxMag = max(max(isoMag), max(sortedFiducial[:, 1]))
    minMag = min(min(isoMag), min(sortedFiducial[:, 1]))

    n = 100
    fMagEval = np.linspace(fiducialLine[1].min(), fiducialLine[1].max(), n)
    fColor = fFid(fMagEval)
    isoFuncShifted = interp1d(
        isoMag, isoColor, bounds_error=False, fill_value="extrapolate"
    )
    iColor = isoFuncShifted(fMagEval)
    dtwX = np.zeros(shape=(n, 1))
    dtwY = np.zeros(shape=(n, 1))

    dtwX[:, 0] = fColor
    dtwY[:, 0] = iColor
    dtwXmask = np.isnan(dtwX).any(axis=1)
    dtwYmask = np.isnan(dtwY).any(axis=1)
    mask = dtwXmask | dtwYmask
    dtwX = dtwX[~mask]
    dtwY = dtwY[~mask]
    if len(dtwX) == 0 | len(dtwY) == 0:
        logger.warning(f"No overlap between isochrone and fiducial line!")
        return np.inf
    if len(dtwX) != len(dtwY):
        logger.warning(f"Shapes do not align! {dtwX.shape} != {dtwY.shape}")
    dtw = fastdtw(dtwX, dtwY, dist=euclidean)
    dists = list()
    for i, j in dtw[1]:
        # compare the distance between the fiducial and isochrone points EEPs
        if ((j < len(iColor) - 1) and j != 0) and ((i < len(fColor) - 1) and i != 0):
            dists.append(
                np.sqrt((fColor[i] - iColor[j]) ** 2 + (fMagEval[i] - fMagEval[j]) ** 2)
            )
    chi2 = np.sum([x**2 for x in dists])
    chi2nu = chi2 / len(dists)

    if verbose:
        logger.info(f"Chi2nu: {chi2nu}, mu: {distance}, Av: {Av}, age: {age}")

    return chi2nu


def optimize(
    fiducial: FARRAY_2D_2C,
    isochrone: Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
    bolTables: Union[str, List[str]],
    FeH: float,
    filters: Tuple[str, str, str] = ("F606W", "F814W", "F606W"),
    verbose: bool = False,
    muSigSize: float = 3,
    muAge: float = 10,
) -> CHI2R:
    """
    Run Chi2 optimization results on photometry and isochrone minimizing the
    age, distance, and reddining needed to fit the isochrone to the fiducial
    line.

    Parameters
    ----------
        fiducialLine : np.ndarray[[np.ndarray[float64], np.ndarray[float64]]]
            Fiducual line in the format output by the fiducual function. Where
            fiducialLine[:, 0] are the colors of each fiducual point and
            fiducialLine[:, 1] are the magniudes of each fiducual point.
        isochrone : dict[pd.DataFrame]
            The isochrone. If loaded and bolometrically corrected using pysep
            then this will be in the correct format. Otherwise make sure that
            it includes filters called "WFC3_UVIS_filter[0..2]_MAG" which have
            your filters in them. In future this will be cleaned up to allow for
            more general filters.
        bolTables : Union[str, List[str]]
            The bolometric correction tables to use. If a list is passed then
            the tables will be read from disk (each list element is a path to
            a table). If a string is passed then the appropriate tables will
            be fetched from the MIST website and used.
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            Filter names to use (which will get injcected to form the column
            names used to get the color and mag). The color is defined as
            filter[0] - filter[1] and the mag is filter[2]
        FeH : float
            The [Fe/H] value to use when bolometrically correcting isochrones.
        verbose : bool, default = False
            Flag controlling whether verbose output will be used. This can
            be helpful for debugging; however, it may slow down the code.
        muSigSize : float, default = 3
            The width of the sigma distribution above and below the central mu
            guess to be explored. Large values will allow for more leway in the
            E(B-V) value optimized. However, they will also slow down the
            optimizer and may increase the likelyhood of falling into a local
            minima
        muAge : float, default=10
            The age at which to estimate the distance modulus in Gyr.

    Returns
    -------
        optimized : OptimizeResult
            The optimization results
    """
    logger = get_logger("fidanka.isofit.fit.optimize")
    logger.info(
        f"Optimizing using [Fe/H] = {FeH}, over {filters} and with muSigSize: {muSigSize}"
    )

    # Do the fiducial interpolation once since it is the same in each
    # iteration
    fFid = interp1d(
        fiducial[1], fiducial[0], bounds_error=False, fill_value="extrapolate"
    )

    # instantiate the bolometric corrector object (this will also
    # allow the FeH interpolation to run only once).
    logger.info("Building Bolometric Corrector Table...")
    bc = BolometricCorrector(bolTables, FeH)
    logger.info("Bolometric Corrector Table Built!")

    logger.info(f"Measuring mu space and {muAge} Gyr...")
    muGuess, muStd = limit_mu_space(
        interp_isochrone_age(isochrone, muAge), fiducial, bc, filters, fFid, pbar=False
    )
    muGuess = abs(muGuess)
    muLower, muUpper = muGuess - 3 * muStd, muGuess + 3 * muStd
    logger.info("Mu space measured!")

    # Wrapper function to allow age to be interpolated on the fly
    age_d_E_opt = lambda iso, age, d, E: get_ISO_CMD_Chi2(
        interp_isochrone_age(iso, age),
        fiducial,
        bc,
        fFid,
        distance=d,
        Av=E,
        filters=filters,
        age=age,
        verbose=verbose,
    )

    # x -> (age, mu, E(B-V))
    objective = lambda x: age_d_E_opt(isochrone, x[0], x[1], x[2])

    logger.info(f"Minimization started with mu: {muLower:0.3f} - {muUpper:0.3f}...")
    optimized = minimize(
        objective,
        [12, abs(muGuess), 0.5],
        bounds=[(5, 20), (muLower, muUpper), (0, 2)],
    )
    logger.info(
        f"Minimization complete, age: {optimized.x[0]:0.3f}, mu: {optimized.x[1]:0.3f}, E(B-V): {optimized.x[2]:0.3f}"
    )

    assert isinstance(optimized, OptimizeResult)
    return {"opt": optimized, "iso": isochrone, "bc": bc}


def order_best_fit_result(
    optimizationResults: dict[str, dict[float, dict[float, CHI2R]]]
) -> dict[str, List[ORT]]:
    """
    Order the best fit optimization results so that they are easy to parse.
    Ordering is done based on the fun attribure of the OptimizeResult object.

    Parameters
    ----------
        optimizationResults : dict[str, dict[float, dict[float, CHI2R]]]
            Where CHI2R is the return type of the optimize function.
            This is a dictionary of those indexed based on population name,
            helium mass fraction, and alpha enhancement.

    Returns
    -------
        comparison : dict[str, List[ORT]]
            Where ORT is a Tuple continuing the chi2nu minimal result, the
            input vector x attaining that result (in the order of age,
            distance, Av), the population name, the helium mass fraction, and
            the alpha enhancement. The dict comparison is broken into two keys,
            one A and one E for the two populations of NGC 2808.

            TODO: Make this more general so that it does not depend on the two
            populations of 2808. This should be pretty trivial to do

    """
    comparison = {"A": list(), "E": list()}
    for pop, popD in optimizationResults.items():
        for Y, YD in popD.items():
            for a, aD in YD.items():
                comparison[pop].append(
                    (aD["opt"]["fun"], aD["opt"]["x"], pop, Y, a, aD["bc"], aD["iso"])
                )
        comparison[pop] = sorted(comparison[pop], key=lambda x: x[0])
    return comparison


def fit_isochrone_to_population(
    fiducialSequences: List[FARRAY_2D_2C],
    ISOs: dict[str, dict[float, dict[float, pd.DataFrame]]],
    filters: Tuple[str, str, str],
    fiducialLookup: dict[str, int],
) -> dict[str, Union[dict[str, ORT], dict[str, dict[float, dict[float, CHI2R]]]]]:
    """
    Take a set of isochrones which vary in population, helium mass fraction,
    and alpha enhancement and fit them to a fiducial line using chi2
    minimization. Then order those isochrones into their best fit version

    Parameters
    ----------
        fiducialSequences : List[np.ndarray[[np.ndarray[float64], np.ndarray[float64]]]]
            list of Fiducual lines in the format output by the fiducual
            function. Where fiducialLine[:, 0] are the colors of each fiducual
            point and fiducialLine[:, 1] are the magniudes of each fiducual
            point. Each element of the list is a differenet fiducual line
            for a differenet population. If you only have one population
            this will be a single element list.
        ISOs : dict[str, dict[float, dict[float, pd.DataFrame]]]
            dictionary of isochrones indexed by population name, helium mass
            fraction, and alpha enhancement. If loaded and bolometrically
            corrected using pysep then this will be in the correct format.
            Otherwise make sure that it includes filters called
            "WFC3_UVIS_filter[0..2]_MAG" which have your filters in them. In
            future this will be cleaned up to allow for more general filters.
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            Filter names to use (which will get injcected to form the column
            names used to get the color and mag. The color is defined as
            filter[0] - filter[1] and the mag is filter[2]
        fiducialLookup : dict[str, int]
            Mapping between ISOs and fiducialSequences. If your ISOs contains
            for example 1 populations called main then fiducial lookup would
            have to be {"main": 0}

    Returns
    -------
        out : dict
            This is a dictionary summarizing the entire minimization process
            the key bf has the orderd optimization results. The key r has the
            more detailed resutls of each chi2 minimization that took place.
            The tuples in bf allow you to lookup anything in r
    """
    results = dict()
    popFid = None
    for population, popISO in tqdm(ISOs.items(), total=len(ISOs), leave=False):
        results[population] = dict()
        popFid = fiducialSequences[fiducialLookup[population]]
        assert popFid is not None
        for Y, YISO in tqdm(popISO.items(), total=len(popISO), leave=False):
            results[population][Y] = dict()
            for a, alphaISO in tqdm(YISO.items(), total=len(YISO), leave=False):
                results[population][Y][a] = optimize(popFid, alphaISO, filters)

    orderedOptimizationResults = order_best_fit_result(results)
    out = {"bf": orderedOptimizationResults, "r": results}
    return out


def shortest_distance_from_point_to_function(x, y, f):
    """
    Computes the shortest distance from a point (x, y) to the curve defined by function f.
    Returns the distance and the closest point on the function.

    Parameters
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    f : callable
        Function of one variable.

    Returns
    -------
    distance : float
        The shortest distance from the point to the curve.
    closest_point : tuple
        The closest point on the curve, as a tuple (x, f(x)).

    """

    def distance(x2):
        return np.sqrt((x2 - x) ** 2 + (f(x2) - y) ** 2)

    result = minimize_scalar(distance)

    return result.fun, (result.x, f(result.x))


def shortest_distance_with_endpoints(f1, f2, domain):
    """
    Compute the pointwise shortest distance and endpoints between two functions over a domain.

    Parameters
    ----------
    f1 : callable
        Function of one variable.
    f2 : callable
        Function of one variable.
    domain : tuple
        Tuple containing (start, end, num_points), which defines the range and discretization.

    Returns
    -------
    distances : ndarray
        Pointwise shortest distances between the two functions over the domain.
    endpoints : list of tuple
        Each tuple contains the starting and ending point of the shortest line segment.

    """

    distances = []
    endpoints = []
    for x in domain:
        dist, endpt = shortest_distance_from_point_to_function(x, f1(x), f2)
        distances.append(dist)
        endpoints.append(((x, f1(x)), endpt))

    return np.array(distances), endpoints


def iterative_objective(
    r,
    bc,
    iso,
    fiducial,
    domain,
    ageChi2=False,
    filters=("WFC3_UVIS_F606W", "WFC3_UVIS_F814W"),
    rFilterOrder=True,
):
    logger = get_logger("")
    mu = r[0]
    Av = r[1]
    age = r[2]

    if Av < 0 or Av > 10 or age < 0 or mu < 0:
        return np.inf
    isoAtAge = interp_isochrone_age(iso, age)
    logger.info(f"mu: {mu}, Av: {Av}, age: {age}, ageChi2: {ageChi2}")

    # Could obtimize this by removing the casting to and from pandas
    header = iso[list(iso.keys())[0]].columns
    isoAtAge = pd.DataFrame(isoAtAge, columns=header)

    corrected = bc.apparent_mags(
        10 ** isoAtAge["log_Teff"].values,
        isoAtAge["log_g"].values,
        isoAtAge["log_L"].values,
        mu=mu,
        Av=Av,
        filters=filters,
    )
    color = corrected[filters[0]] - corrected[filters[1]]
    if rFilterOrder:
        mag = corrected[filters[1]]
    else:
        mag = corrected[filters[0]]
    isoF = interp1d(mag, color, bounds_error=False, fill_value="extrapolate")
    shortest_distances, line_endpoints = shortest_distance_with_endpoints(
        fiducial, isoF, domain
    )
    if ageChi2:
        chi2 = (
            np.sqrt(np.sum(shortest_distances))
            + (np.max(shortest_distances) - np.min(shortest_distances))
        ) * np.std(shortest_distances)
    else:
        chi2 = np.sqrt(np.sum(shortest_distances)) / shortest_distances.shape[0]
    logger.info(f"Chi2 {chi2}")
    return chi2


def iterative_optimize(
    bounds,
    iso,
    fiducial,
    domain,
    bc,
    filters=("WFC3_UVIS_F275W", "WFC3_UVIS_F814W"),
    rFilterOrder=True,
    getChi2Dist=False,
):
    logger = get_logger("iterative_optimize")
    initGuesses = [(x[0] + x[1]) / 2 for x in bounds]
    outputDist = [[[], []], [[], []], [[], []]]

    def chi2_wrapper(x, o, vID):
        chi2 = o(x)
        x = x[0]
        if getChi2Dist:
            print(f"Appending {x} to {vID} with chi2 {chi2}")
            outputDist[vID][0].append(x)
            outputDist[vID][1].append(chi2)
        return chi2

    for vID, (bound, guess) in enumerate(zip(bounds, initGuesses)):
        logger.info(f"Optimizing {vID} with bounds {bound}")
        o = lambda x: iterative_objective(
            [ig if ID != vID else x[0] for ID, ig in enumerate(initGuesses)],
            bc,
            iso,
            fiducial,
            domain,
            ageChi2=True if vID == 2 else False,
            filters=filters,
            rFilterOrder=rFilterOrder,
        )
        rGuess = np.random.uniform(low=bound[0], high=bound[1], size=1)[0]
        r = minimize(
            lambda x: chi2_wrapper(x, o, vID), bound[0], bounds=[(bound[0], None)]
        )
        initGuesses[vID] = r.x[0]
    logger.info(f"Initial guesses: {initGuesses}")
    r = minimize(
        lambda r: iterative_objective(
            r,
            bc,
            iso,
            fiducial,
            domain,
            ageChi2=False,
            filters=("WFC3_UVIS_F275W", "WFC3_UVIS_F814W"),
            rFilterOrder=True,
        ),
        initGuesses,
        tol=0.002,
        bounds=([5, None], [0.001, 1], [1, 25]),
    )
    logger.info(f"Final Optimized Parameters: {r.x}")
    output = (r, [np.array(x) for x in outputDist]) if getChi2Dist else r
    return output


def parallel_optimize(
    bounds,
    isopath_list,
    fiducialLine,
    bc,
    filters=("WFC3_UVIS_F275W", "WFC3_UVIS_F814W"),
    rFilterOrder=True,
    getChi2Dist=False,
):
    iso_list = [read_iso(path) for path in isopath_list]
    logger = get_logger("parralel optimization")
    results = []

    logger.info("Spinning up process pool")
    fFunc = interp1d(
        fiducialLine.mean[1],
        fiducialLine.mean[0],
        bounds_error=False,
        fill_value="extrapolate",
    )
    domain = np.arange(np.min(fiducialLine.mean[1]), np.max(fiducialLine.mean[1]), 0.05)
    with ProcessPoolExecutor() as executor, tqdm(total=len(iso_list)) as pbar:
        futures = {
            executor.submit(
                iterative_optimize,
                bounds,
                iso,
                fFunc,
                domain,
                bc,
                filters,
                rFilterOrder,
                getChi2Dist,
            ): (iso, path)
            for iso, path in zip(iso_list, isopath_list)
        }

        for future in as_completed(futures):
            logger.info("Completed optimization for isochrone, collecting result")
            try:
                result = future.result()
                print(result)
                exit()
                results.append((futures[future][1], result))
                logger.info("Completed Optimization for isochrone.")
            except Exception as e:
                results.append((futures[future][1], None))
                logger.error(
                    f"Error Processing isochrone {futures[future][1]}. Exception {e}"
                )

            # Update progress bar
            pbar.update(1)
    logger.info("Finished optimization")

    return results


if __name__ == "__main__":
    import logging
    import pickle as pkl

    with open(
        "../../../../../GraduateSchool/Thesis/GCConsistency/NGC2808/Analysis/fiducial/ngc2808fiducials.pkl",
        "rb",
    ) as f:
        fiducial = pkl.load(f)
    bounds = [[5, 20], [0.001, 1], [7, 20]]
    import pathlib

    isoList = list(
        pathlib.Path(
            "../../../../../GraduateSchool/Thesis/GCConsistency/NGC2808/outputs.denseAlpha.fixedLowmass/PopA+0.27"
        ).rglob("isochrones.txt")
    )[:3]

    # isoList = [read_iso(path) for path in isoList]

    from fidanka.bolometric.bctab import BolometricCorrector

    bc = BolometricCorrector("WFC3", -0.9)

    results = parallel_optimize(
        bounds,
        isoList,
        fiducial[1],
        bc,
        filters=("WFC3_UVIS_F275W", "WFC3_UVIS_F814W"),
        rFilterOrder=True,
    )
    print(results)
