from fidanka.isochrone.isochrone import shift_isochrone
from fidanka.isochrone.isochrone import interp_isochrone_age
from fidanka.fiducial.fiducial import fiducial_line

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Callable, Tuple, Union, List, Any

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult

from tqdm import tqdm
import pickle as pkl

FARRAY_1D = npt.NDArray[np.float64]
R2_VECTOR = npt.NDArray[[np.float64, np.float64]]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]

CHI2R = dict[str, Any]
ORT = Tuple[float, Tuple[float, float, float], str, float, float]

def pfD(
        r : R2_VECTOR,
        I : Callable
        ) -> Callable:
    """
    Return a function which givex the perpendicular distance between a point and
    a function evaluated at some point x

    Parameters
    ----------
        r : np.ndarray[float64]
            2-vector (x,y), some point
        I : Callable
            Function of a single parameter I(x) -> y.

    Returns
    -------
        d : Callable
            Function of form d(x) which gives the distance between r and I(x)
    """
    return lambda m: np.sqrt((I(m) - r[0])**2 + (m - r[1])**2)

def get_synthPop_chi2(
        observedFiducial : FARRAY_2D_2C,
        theoreticalPop : FARRAY_2D_2C,
        distance : float = 0,
        E_BV : float = 0,
        reversedFilterOrder : bool = False,
        reduced : bool = True,
        ) -> float:
    theoreticalFiducual = fiducial_line(theoreticalPop[:,0], theoreticalPop[:,1],
                                        theoreticalPop[:,2], theoreticalPop[:,3],
                                        reverseFilterOrder=reversedFilterOrder,
                                        mcruns=1)

    modelColor = theoreticalFiducual[:,0] - theoreticalFiducual[:,1]
    if reversedFilterOrder:
        modelMag = theoreticalFiducual[:,1]
    else:
        modelMag = theoreticalFiducual[:,0]
    modelAptColor, modelAptMag = shift_isochrone(modelColor, modelMag, distance, E_BV)

    f = interp1d(
            modelAptMag,
            modelAptColor,
            bounds_error=False,
            fill_value='extrapolate'
            )

    diffs = observedFiducial[:, 0] - f(observedFiducial[:, 1])
    sortedFiducial = observedFiducial[observedFiducial[:,1].argsort()]
    minDist = np.empty(shape=(sortedFiducial.shape[0]))
    minDist[:] = np.nan
    for idx, point in enumerate(sortedFiducial):
        d = pfD(point, f)
        nearest = minimize(d, point[1], method='Nelder-Mead')
        if not nearest.success:
            print(f"Warning: {nearest.message}")
        else:
            minDist[idx] = d(nearest.x)
    minDist = minDist[~np.isnan(minDist)]
    chi2 = np.sum(np.apply_along_axis(np.square, 0, minDist))
    if reduced:
        chi2 = chi2 / minDist.shape[0]
    return chi2


def get_ISO_CMD_Chi2(
        iso : pd.DataFrame,
        fiducialLine : FARRAY_2D_2C ,
        filters : Tuple[str, str, str] = ("F606W", "F814W", "F606W"),
        distance : float = 0,
        Av : float = 0,
        ) -> float:
    """
    Calculate the reduced chi2 value between an isochrone and a fiducialLine
    at a given distance in parsecs and color excess. The chi2 value is 
    calculated as the sum of the square of the minimal perpendicular distances
    between the fiducual line and the isochrone. 

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
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            Filter names to use (which will get injcected to form the column
            names used to get the color and mag. The color is defined as 
            filter[0] - filter[1] and the mag is filter[2]
        distance : float, default = 0
            Distance in parsecs to shift isochrone by when calculating the chi2
            value
        Av : float, default = 0
            Color excess to shift isochrone by when calculating the chi2 value

    Returns
    -------
        chi2nu : float
            Reduced chi2 value. Calculated from the sum of the squares of the
            minimal perpendicular distance between the fiducual line and
            isochrone. Chi2 reduction is preformed then by dividing the chi2
            value by the number of points used to calculate it.
    """
    isoFilter1Name = f"WFC3_UVIS_{filters[0]}_MAG"
    isoFilter2Name = f"WFC3_UVIS_{filters[1]}_MAG"
    isoFilter3Name = f"WFC3_UVIS_{filters[2]}_MAG"

    isoColor = iso[isoFilter1Name] - iso[isoFilter2Name]
    isoMag = iso[isoFilter3Name]

    assert isoColor is not None
    assert isoMag is not None

    assert isinstance(isoMag, pd.Series)
    assert isinstance(isoColor, pd.Series)


    isoAptMag, isoAptColor = shift_isochrone(isoColor, isoMag, distance, Av)

    f = interp1d(
            isoAptMag,
            isoAptColor,
            bounds_error=False,
            fill_value='extrapolate'
            )

    diffs = fiducialLine[:, 0] - f(fiducialLine[:, 1])
    sortedFiducial = fiducialLine[fiducialLine[:,1].argsort()]
    minDist = np.empty(shape=(sortedFiducial.shape[0]))
    minDist[:] = np.nan

    # For each point in the fiducual line find the minimum possible distance
    # to the isochrone
    for idx, (point, _) in enumerate(zip(sortedFiducial, diffs)):
        d = pfD(point, f)
        nearestPoint = minimize(d, 0, method='Nelder-Mead')
        if not nearestPoint.success:
            print("FAIL")
        else:
            minDist[idx] = d(nearestPoint.x[0])

    minDist = minDist[~np.isnan(minDist)]
    chi2 = np.sum(np.apply_along_axis(lambda x: x**2,0,minDist))
    chi2nu = chi2/minDist.shape[0]
    return chi2nu

def optimize(
        fiducial : FARRAY_2D_2C,
        isochrone : pd.DataFrame,
        filters : Tuple[str, str, str]
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
        isochrone : pd.DataFrame
            The isochrone. If loaded and bolometrically corrected using pysep
            then this will be in the correct format. Otherwise make sure that 
            it includes filters called "WFC3_UVIS_filter[0..2]_MAG" which have
            your filters in them. In future this will be cleaned up to allow for
            more general filters.
        filters : Tuple[str, str, str], default=("F606W", "F814W", "F606W")
            Filter names to use (which will get injcected to form the column
            names used to get the color and mag. The color is defined as 
            filter[0] - filter[1] and the mag is filter[2]
    """
    age_d_E_opt = lambda iso, age, d, E: get_ISO_CMD_Chi2(
            interp_isochrone_age(iso, age),
            fiducial,
            distance=d,
            Av=E,
            filters=filters)

    optimized = minimize(
            lambda x: age_d_E_opt(isochrone, x[0], x[1], x[2]),
            [12, 10000, 0.06],
            bounds=[
                (5,20),
                (5000, None),
                (0,0.3)
                ],
            )

    assert isinstance(optimized, OptimizeResult)

    return {'opt': optimized, 'iso': isochrone, 'fiducial': fiducial}

def optimize_theoretical_population(


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
    comparison = {'A': list(), 'E': list()}
    for pop, popD in optimizationResults.items():
        for Y, YD in popD.items():
            for a, aD in YD.items():
                comparison[pop].append((aD['opt']['fun'], aD['opt']['x'], pop, Y, a))
        comparison[pop] = sorted(comparison[pop], key=lambda x: x[0])
    return comparison

def fit_isochrone_to_population(
        fiducialSequences : List[FARRAY_2D_2C],
        ISOs : dict[str, dict[float, dict[float, pd.DataFrame]]],
        filters : Tuple[str, str, str],
        fiducialLookup : dict[str, int]
        ) -> dict[str, Union[
            dict[str, ORT],
            dict[str, dict[float, dict[float, CHI2R]]]
            ]]:
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
    out = {'bf': orderedOptimizationResults, 'r': results}
    return out
