from fidanka.exception.exception import shape_dimension_check
from fidanka.warn.warnings import warning_traceback
# from fidanka.ext.nearest_neighbors import nearest_neighbors

import numpy as np
import pandas as pd
import os

import numpy.typing as npt
from typing import Union, Tuple, Callable

from scipy.optimize import curve_fit, fsolve, newton, brentq, ridder
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import splrep, BSpline, interp1d
from scipy.spatial._qhull import ConvexHull as ConvexHullType
from scipy.stats import norm

from tqdm import tqdm

import warnings
import hashlib

import logging
import sys

IARRAY_1D = npt.NDArray[np.int32]

FARRAY_1D = npt.NDArray[np.float64]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]
FARRAY_2D_3C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D]]
FARRAY_2D_4C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D, FARRAY_1D]]

R2_VECTOR = npt.NDArray[[np.float64, np.float64]]


def ridge_bounding(
        a1 : FARRAY_1D,
        a2 : FARRAY_1D,
        binsLeft : FARRAY_1D,
        histBins : Union[int, str]=50,
        allowMax : bool=False,
        ) -> FARRAY_2D_4C:
    """
    Find a ridge line for some data x=a1, y=a2 binned along a2 in bins whos
    left (bottom) edges are defined in binsLeft.

    This algorithm works by slicing a1 and a2 by which points fall within each
    bin then fitting a unimodal normal distribution to a histogram of those
    point along the a1 axis.

    Addotionally the 4th and 96th percentiles along the a1 axis are returned for
    each bin in a2

    Parameters
    ----------
        a1 : ndarray[float64]
            Axis 1, x coordinates of each data point
        a2 : ndarray[float64]
            Axis 2, y coordinates of each data point
        binsLeft : ndarray[float64]
            left (bottom) edges of bins along a2 axis. Bin size will be
            calculated from binsLeft[1]-binsLeft[0]
        histBins : Union[int,str], default=50
            Number of bins to use when generating the histogram. See numpy
            histogram documentation for valid stings to use.
        allowMax : bool, default=False
            If true, if the fitting fails for a given bin then the bin will be
            filled with the max value of the histogram. If false, the bin will
            be filled with nans.

    Returns
    -------
        ridgeLine : ndarray[[ndarray[float64], ndarray[float64]]]
            Ridge line fit to a1 and a2. ridgeLine[0, :] gives the coordinates
            of the ridge line along the a1 axis, while ridgeLine[1, :] gives
            the coordinates of the ridgeLine along the a2 axis (these
            are the infered centers points of the bins)

    Examples
    --------
    Given some array color and magnitude, defining some CMD for a globular
    clusters then you could find an approximate ridge line to that CMD using

    >>> from fidanka.fiducial import ridge_bounding
    >>> import numpy as np
    >>>
    >>> magRange = (np.percentile(magnitude, 5), np.percentile(magnitude, 95))
    >>> binsLeft = np.arange(magRange[0], magRange[1], 0.1)
    >>> ridgeLine = ridge_bounding(color, magnitude, binsLeft)

    Note how the mag range does not include all the data. This is to avoid
    fitting issuse at in the lowest counting statistics portions of the CMD.
    """
    shape_dimension_check(a1, a2)

    gaus = lambda x, a, b, c: a*np.exp(-(x-b)**2/(2*c**2))
    binSize = binsLeft[1] - binsLeft[0]
    binsRight = binsLeft + binSize
    ridgeLine = np.zeros(shape=(4,binsLeft.shape[0]))
    for binID, (left, right) in enumerate(zip(binsLeft, binsRight)):
        cut = (a2 >= left) & (a2 < right)
        Xss = a1[cut]
        histV, histLE = np.histogram(Xss,bins=histBins)
        histC = (histLE + (histLE[1] - histLE[0])/2)[:-1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = curve_fit(
                        gaus,
                        histC,
                        histV,
                        p0=[max(histV),histC[histV.argmax()],0.01]
                        )
            ridgeLine[0,binID] = fit[0][1]
            ridgeLine[1,binID] = (left+right)/2
            ridgeLine[2,binID] = np.percentile(Xss,4)
            ridgeLine[3,binID] = np.percentile(Xss,96)
        except RuntimeError:
            lenOkay = cut[cut == True].shape[0]
            print(f"Unable to fit for bin {binID} from {left}->{right} with {lenOkay} targets")
            if len(histV) > 1 and allowMax:
                print("Falling back to max value")
                ridgeLine[:, binID] = histC[np.argmax(histV)]
            elif len(histV) > 1 and not allowMax:
                print("Falling back to nan")
                ridgeLine[:, binID] = np.nan
    return ridgeLine[:, 1:-1]

def instantaious_hull_density_cpp(
        r0: R2_VECTOR,
        ri: FARRAY_2D_2C,
        n : int=100
        ) -> Tuple[np.float64, ConvexHullType, IARRAY_1D]:
    """
    Calculate the density at a given point in a dataset in a way which keeps
    the counting statistics for each density calculation uniform. In order to
    do this four steps.

        1. Calculate the euclidian distance between the point in question (ri)
        and every other point (r0[j]) in the data set.
        2. Partition the 50 smallest elements from the calculated distance array
        3. Use those indicies to select the 50 points from r0 which are the
        closest to ri
        4. Find the convex hull of those 50 (it will be 50 not 51 as r0 is
        assumed to be a member of ri so there will always be at least one
        distance of 0 in the distance array which is the self distance)
        5. Define and return the area as the number of points (n) / the area of
        the convex hull

    This method dynamically varies the area considered when calculating the
    density in order to maintain a fixed number of samples per area bin.

    Parameters
    ----------
        r0 : ndarray[[float64, float64]]
            x, y coordinates of the point where you wish to calculate the
            density of. Assumed to be a member of ri. If this is not a member
            of ri the algorithm should still work under the constraint that the
            data is well distributed in all directions around r0, otherwise the
            area calculation could be off.
        ri : ndarray[[ndarray[float64], ndarray[float64]]]
            x,y coordinates of all data points. This should have the shape
            (n, 2) where n is the numbe of data points
        n : int, default=100
            Number of closest points to considered when finding the convex hull
            used to define the area.

    Returns
    -------
        density : float64
            The approximate density at ri embeded within the r0 data field
        hull : ConvexHull
            The fully computed convex hull of the n closest points to ri within
            the ri data field. Computed by scipy.spatial.ConvexHull (qhull)
        partition : ndarray[int]
            The indicies of the n closest points to ri within the ri data field.
    """
    partition = nearest_neighbors(r0, ri, n)
    hullPoints = ri[partition]
    hull = ConvexHull(hullPoints)
    density = hullPoints.shape[0]/hull.volume

    return density, hull, partition

def instantaious_hull_density(
        r0: R2_VECTOR,
        ri: FARRAY_2D_2C,
        n : int=100
        ) -> Tuple[np.float64, ConvexHullType, IARRAY_1D]:
    """
    Calculate the density at a given point in a dataset in a way which keeps
    the counting statistics for each density calculation uniform. In order to
    do this four steps.

        1. Calculate the euclidian distance between the point in question (ri)
        and every other point (r0[j]) in the data set.
        2. Partition the 50 smallest elements from the calculated distance array
        3. Use those indicies to select the 50 points from r0 which are the
        closest to ri
        4. Find the convex hull of those 50 (it will be 50 not 51 as r0 is
        assumed to be a member of ri so there will always be at least one
        distance of 0 in the distance array which is the self distance)
        5. Define and return the area as the number of points (n) / the area of
        the convex hull

    This method dynamically varies the area considered when calculating the
    density in order to maintain a fixed number of samples per area bin.

    Parameters
    ----------
        r0 : ndarray[[float64, float64]]
            x, y coordinates of the point where you wish to calculate the
            density of. Assumed to be a member of ri. If this is not a member
            of ri the algorithm should still work under the constraint that the
            data is well distributed in all directions around r0, otherwise the
            area calculation could be off.
        ri : ndarray[[ndarray[float64], ndarray[float64]]]
            x,y coordinates of all data points. This should have the shape
            (n, 2) where n is the numbe of data points
        n : int, default=100
            Number of closest points to considered when finding the convex hull
            used to define the area.

    Returns
    -------
        density : float64
            The approximate density at ri embeded within the r0 data field
        hull : ConvexHull
            The fully computed convex hull of the n closest points to ri within
            the ri data field. Computed by scipy.spatial.ConvexHull (qhull)
        partition : ndarray[int]
            The indicies of the n closest points to ri within the ri data field.
    """
    distance = cdist(r0.reshape(1,2), ri)[0]
    partition = np.argpartition(distance, n)[:n]
    hullPoints = ri[partition]
    hull = ConvexHull(hullPoints)
    density = hullPoints.shape[0]/hull.volume

    return density, hull, partition


def hull_density(

        X: FARRAY_1D,
        Y : FARRAY_1D,
        n : int = 100
        ) -> FARRAY_1D:
    """
    Calculate the number density at each point (i, j) for i, j in X, Y using
    the instantaious_hull_density function. X and Y must have the same shape

    Parameters
    ----------
        X : ndarray[float64]
            x values of dataset
        Y : ndarray[float64]
            y values of dataset
        n : int, default=100
            Number of closest points to considered when finding the convex hull
            used to define the area in the instantaious_hull_density function.

    Returns
    -------
        density : ndarray[float64]
            1D array of density. If len(X) == len(Y)

    Examples
    --------
    Given some array color and magnitude, defining some CMD for a globular
    clusters then you could find the approximate density at each target

    >>> from fidanka.fiducial import hull_density
    >>> density = hull_density(color, magnitude)

    Note that this is not a very fast algorithm as its complexity scales
    like O(n^2). For each data point the distance to every other data point
    must be calculated.
    """
    shape_dimension_check(X, Y, dim=1)

    r = np.vstack((X, Y)).T
    density = np.apply_along_axis(
            lambda r0: instantaious_hull_density(r0, r, n=n)[0],
            1,
            r
            )
    density = density/np.median(density)
    return density

def color_mag_from_filters(
        filter1 : Union[FARRAY_1D, pd.Series],
        filter2 : Union[FARRAY_1D, pd.Series],
        reverseFilterOrder : bool
        ) -> Tuple[FARRAY_1D, FARRAY_1D]:
    """
    Given two filters and the a flag saying which one to use as the magnitude
    return the color and magnitude arrays.

    Parameters
    ----------
        filter1 : Union[ndarray[float64], pd.Series]
            First filter, will be A in A-B color
        filter2 : Union[ndarray[float64], pd.Series]
            Second filter, will be B in A-B color
        reverseFilterOrder : bool
            Flag to determine which filter is used as the magnitude. If
            reverseFilterOrder is false then filter1 is is the magnitude. If
            reverseFilterOrder is True then filter2 is the magnitude

    Returns
    -------
        Tuple[color, magnitude]
            Where color and magnitude are of type
            Union[ndarray[float64], pd.Series] and have the same shape.

    """
    shape_dimension_check(filter1, filter2, dim=1)

    color = filter1-filter2
    mag = filter2 if reverseFilterOrder else filter1

    if isinstance(color, pd.Series):
        color = color.values
    if isinstance(mag, pd.Series):
        mag = mag.values

    assert isinstance(color, np.ndarray)
    assert isinstance(mag, np.ndarray)

    return color, mag

def shift_photometry_by_error(
        phot : Union[FARRAY_1D, pd.Series],
        err : Union[FARRAY_1D, pd.Series],
        baseSampling : FARRAY_1D,
        ) -> FARRAY_1D:
    """
    Shift each point, p_i, in phot by some amount sampled from a normal
    distribution with standard deviation equal to its associated uncertainty,
    e_i, in err. Phot and Err must both be 1D and must both be the same shape

    Parameters
    ----------
        phot : Union[ndarray[float64], pd.Series]
            Nominal values of Photometry (single filter)
        err : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry.
        baseSampling : ndarray[float64]
            Random samples from a normal distribution with mu = 0 and sig = 1
            which will be rescaled to the correct standard deviation and
            used to shift the photometry by the error. Must have the same
            shape as filter1 and filter2.

    Returns
    -------
        shiftedPhot : ndarray[float64]
            Array of the same shape as phot but with each point shifted
            by some amount sampled from a normal distribution with standard
            deviation equal to that of the uncertainty of each point.
    """
    shape_dimension_check(phot, err)

    shiftedPhot = phot + np.multiply(baseSampling, err)

    return shiftedPhot

def MC_convex_hull_density_approximation(
        filter1 : Union[FARRAY_1D, pd.Series],
        filter2 : Union[FARRAY_1D, pd.Series],
        error1 : Union[FARRAY_1D, pd.Series],
        error2 : Union[FARRAY_1D, pd.Series],
        reverseFilterOrder : bool = False,
        mcruns : int = 10,
        convexHullPoints : int = 100,
        pbar : bool = True,
        uni_density: bool = False,
        ) -> FARRAY_1D:
    """
    Compute the density at each point in a CMD accounting for uncertainty
    in data. Uncertainty is accounted for through monte carlo sampleling wherin
    all data is shifted around based on individual uncertainties, the density
    is calculate using the methods described in the hull_density function, and
    then process is repeated mcruns times. After each repetion the density is
    averaged with a running density. The final reported density at each point
    is the average density at each point with mcruns shifts of the data around
    based on its photometric uncertainty.

    Parameters
    ----------
        filter1 : Union[ndarray[float64], pd.Series]
            First filter, will be A in A-B color
        filter2 : Union[ndarray[float64], pd.Series]
            Second filter, will be B in A-B color
        error1 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter1.
        error2 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter2.
        reverseFilterOrder : bool, default=False
            Flag to determine which filter is used as the magnitude. If
            reverseFilterOrder is false then filter1 is is the magnitude. If
            reverseFilterOrder is True then filter2 is the magnitude
        mcruns : int, default=10
            Number of monte carlo runs to use when calculating the density. Note
            that increasing this will linearlly slow down your code. If
            mcruns is set to 1 then the density will be calculated without
            accounting for uncertainty.
        convexHullPoints : int, default=100
            Number of closest points to considered when finding the convex hull
            used to define the area in the instantaious_hull_density function.
        pbar : bool, default=True
            Flag controlling whether a progress bar is written to standard output.
            This will marginally slow down your code; however, its a very small
            effect and I generally find it helpful to have this turned on.
        uni_density : bool, default=False
            Flag controls whether to change the sampling method to achieve a roughly
            uniform distribution of numbers of data points in filter1 magnitude

    Returns
    -------
        density : ndarray[float64]
            1D array of density at each point in the color magnitude diagram.

    Examples
    --------
    Given some pandas dataframe which contains photometric data called
    "Photometry" the density of the F275W-F814W CMD could be computed

    >>> from fidanka.fiducial import MC_convex_hull_density_approximation
    >>> f1 = Photometry["F275W"]
    >>> f2 = Photometry["F814W"]
    >>> density = MC_convex_hull_density_approximation(f1, f2, reverseFilterOrder=True)

    """

    if mcruns > 1:
        shape_dimension_check(filter1, error1)
        shape_dimension_check(filter2, error2)
    shape_dimension_check(filter1, filter2)

    density = np.empty_like(filter1)
    if uni_density == True:
        filter1, error1, filter2, error2 = renormalize(filter1,error1,filter2,error2)
        mcrun = mcrun//10 + 1
    if mcruns > 1:
        baseSampling = np.random.default_rng().normal(size=(mcruns, 2, error1.shape[0]))
    else:
        baseSampling = np.zeros(shape=(1, 2, filter1.shape[0]))
        pbar = False

    for i, bs in tqdm(enumerate(baseSampling), disable=not pbar, desc="Monte Carlo Density", total=baseSampling.shape[0]):
        if mcruns > 1:
            f1s = shift_photometry_by_error(filter1, error1, bs[0])
            f2s = shift_photometry_by_error(filter2, error2, bs[1])
            colorS, magS = color_mag_from_filters(f1s, f2s, reverseFilterOrder)
        else:
            colorS, magS = color_mag_from_filters(filter1, filter2, reverseFilterOrder)

        tDensity = hull_density(colorS, magS, n=convexHullPoints)

        density[:] = tDensity if i == 0 else (density[:] + tDensity[:])/2



    return density

def renormalize(
        filter1 : Union[FARRAY_1D, pd.Series],
        filter2 : Union[FARRAY_1D, pd.Series],
        error1 : Union[FARRAY_1D, pd.Series],
        error2 : Union[FARRAY_1D, pd.Series],
        ) -> Tuple[FARRAY_1D, FARRAY_1D, FARRAY_1D, FARRAY_1D]:
    """
    Resample the data in the first place to ensure new data has roughly uniform
    distribution in filter1 magnitude. To achieve that, data will be segmented by
    filter1 magnitude. The uniformity is achieved by over sample low density region
    and undersample high density region.
    This function is only used when uni_density = True

    Parameters
    ----------
        filter1 : Union[ndarray[float64], pd.Series]
            First filter, will be A in A-B color
        filter2 : Union[ndarray[float64], pd.Series]
            Second filter, will be B in A-B color
        error1 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter1.
        error2 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter2.

    Returns
    -------
        filter1_renomalized : Union[ndarray[float64], pd.Series]
            Renomalized first filter, will be A in A-B color
        filter2_renomalized : Union[ndarray[float64], pd.Series]
            Renomalized second filter, will be B in A-B color
        error1_renomalized : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from renomalized filter1.
        error2_renomalized : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from renomalized filter2.
    """

    df = pd.DataFrame(data={'filter1': filter1, 'error1': error1, 'filter2': filter2, 'error2': error2})
    num_seg = 100
    seg = np.linspace(min(df['filter1']),max(df['filter1'])+0.0001,num_seg+1)
    df_seg = [df[(df['filter1'] >= seg[i]) & (df['filter1'] < seg[i+1])] for i in range(num_seg)]
    data_count_min = min([len(df) for df in df_seg])

    while data_count_min < 5:
        num_seg -= 1
        seg = np.linspace(min(df['filter1']),max(df['filter1'])+0.0001,num_seg+1)
        df_seg = [df[(df['filter1'] >= seg[i]) & (df['filter1'] < seg[i+1])] for i in range(num_seg)]
        data_count_min = min([len(df) for df in df_seg])
    target_num = max([len(df) for df in df_seg])*10

    for i in range(num_seg):
        df_seg[i] = df_seg[i].iloc[np.random.randint(0,high=len(df_seg[i]),size=target_num)]
    df = pd.concat(df_seg)

    return df['filter1'].values, df['error1'].values, df['filter2'].values, df['error2'].values





def percentile_range(
        X : Union[FARRAY_1D, pd.Series],
        percLow : float,
        percHigh : float
        ) -> Tuple[float, float]:
    """
    Extract ranges from an array based on requested percentiles

    Parameters
    ----------
        X : Union[ndarray[float64], pd.Series]
            Array to extract range from
        percLow : float
            Lower bound percentile to base range on
        percHigh : float
            Upper bound percentile to base range on

    Returns
    -------
        xRange : Tuple[float, float]
            range of X between its percLow and percHigh percentiles.
    """
    xRange = (np.percentile(X, percLow), np.percentile(X, percHigh))
    return xRange

def get_mag_and_color_ranges(
        color : Union[FARRAY_1D, pd.Series],
        mag : Union[FARRAY_1D, pd.Series],
        percLow : float,
        percHigh : float
        ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Find the ranges of both color and magnitude between their percLow and
    percHigh percentiles

    Parameters
    ----------
        color : Union[ndarray[float64], pd.Series]
            1D array of color of shape m
        mag : Union[ndarray[float64], pd.Series]
            1D array of magnitude of shape m
        percLow : float
            Lower bound percentile to base range on
        percHigh : float
            Upper bound percentile to base range on

    Returns
    -------
        colorRange : Tuple[float, float]
            Range of color between its percLow and percHigh percentiles
        magRange : Tuple[float, float]
            Range of mag between its percLow and percHigh percentiles
    """
    colorRange = percentile_range(color, percLow, percHigh)
    magRange = percentile_range(mag, percLow, percHigh)
    return colorRange, magRange

def mag_bins(
        mag : Union[FARRAY_1D, pd.Series],
        percHigh : float,
        percLow : float,
        binSize : float
        ) -> Tuple[FARRAY_1D, FARRAY_1D]:
    """
    Find the left and right edges of bins in magnitude space between magnitudes
    percLow and percHigh percentiles with a bin size of binSize.

    Parameters
    ----------
        mag : Union[ndarray[float64], pd.Series]
            1D array of magnitude of shape m
        percLow : float
            Lower bound percentile to base range on
        percHigh : float
            Upper bound percentile to base range on
        binSize : float
            Spacing between each left bin edge to each right bin edge

    Returns
    -------
        binsLeft : ndarray[float64]
            left edges of bins in magnitude space
        binsRight : ndarray[float64]
            right edges of bins in magnitude space
    """
    magRange = percentile_range(mag, percLow, percHigh)
    binsLeft = np.arange(magRange[1], magRange[0], binSize)
    binsRight = binsLeft + binSize
    return binsLeft, binsRight

def density_color_cut(
        density : FARRAY_1D,
        color : FARRAY_1D,
        mag : FARRAY_1D,
        left : float,
        right : float,
        smooth : bool = True
        ) -> Tuple[FARRAY_1D, FARRAY_1D]:
    """
    Cut, sort by color, and potentially smooth density and color into magnitude
    bins.

    Parameters
    ----------
        density : ndarray[float64]
            Density at each point in a CMD of shape m.
        color : ndarray[float64]
            Color of each target in a CMD of shape m.
        mag : ndarray[float64]
            Magnitude of each target in a CMD of shape m.
        left : float
            Magnitude value of the left bin edge. The left edge is inclusive.
        right : float
            Magnitude value of the right bin edge. The right edge is non
            inclusive
        smooth : bool, default=True
            Flah to control whether a savgol_filter of order 2 and window size
            50 is applied to the density data after it is sorted by ascending
            color.

    Returns
    -------
        densityCut : ndarray[float64]
            Density of all points within the magnitude window [left,right)
            ordered by ascending color of points. If smooth is turned on then
            this will be run through a savgol_filter before returning.
        colorCut : ndarray[float64]
            Color of all points within the magnitude window [left, right)
            ordered by ascending color of points.
    """
    colorCut = color[(mag >= left) & (mag < right)]
    densityCut = density[(mag >= left) & (mag < right)]

    densityCut = densityCut[colorCut.argsort()]
    colorCut = colorCut[colorCut.argsort()]

    if smooth and len(densityCut) > 50:
        densityCut = savgol_filter(densityCut, 50, 2)
    return densityCut, colorCut

def noise_robust_spline_peak_extraction(
        color : FARRAY_1D,
        density : FARRAY_1D,
        smin : float,
        smax : float,
        sn : int
        ) -> Tuple[FARRAY_1D, FARRAY_1D]:
    """
    Extract the peaks of sn cubic bivariate bsplines fit to density vs. color.
    Each spline will have a different smoothing factor s which are iterated
    over and defined in np.linspace(smin, smax, sn). The principal with this is
    that the higher the smoothing factor the more noise will be smeared over.
    However, too high and true peaks will be smoothed over.

    Given that we do not have a priori knowledge of which smoothing factor is
    ideal for a given density profile / population we would like to avoid
    selecting a fixed smoothing factor. Instead we could exploit the property
    that peaks from noise should smooth out before true peaks.

    What we could do is fit sn splines each with a slightly larger smoothing
    factor than the pervious spline. For each spline we then use a peak
    detection algorithm to identify peaks and store all of them in an array.
    This function only takes care of this part. However, continue reading for
    a full explination of the algorithm in use.

    True peaks should be detected more often than peaks from noise therefore
    in true peaks should dominate in the distribution of peak colors.

    Parameters
    ----------
        color : ndarray[float64]
            Color of each target in a CMD of shape m.
        density : ndarray[float64]
            Density at each point in a CMD of shape m.
        smin : float
            Miniumum smoothing factor to use. Generally this should be small
            but non zero (0.1 has worked well for me)
        smax : float
            Maximum smoothing factor to use. I find that its not good to push
            this much past 1
        sn : int
            Overall number of splines to fit. The spline fitting routines in
            scipy, which this function makes use of, are well optimized so
            you can push this pretty high without too many preformance issues.
            I find that a value of 100-200 works well. However, this will depend
            on your specific density profile.

    Returns
    -------
        spp : ndarray[float64]
            1D array of all the colors of all peaks extracted from every spline
            fit to the data
        sdd : ndarray[float64]
            1D array of the densities (heights) of each peak extracted from
            every spline fit to the data
    """
    srange = np.linspace(smin, smax, sn)
    sp, sd = list(), list()
    cleanColor = color[~np.isnan(color)]
    cleanDensity = density[~np.isnan(color)]

    for s in srange:
        try:
            tck = splrep(cleanColor, cleanDensity, s=s)
            splineVals = BSpline(*tck)(cleanColor)
            peaks, _ = find_peaks(splineVals)

            sp.append(cleanColor[peaks])
            sd.append(cleanDensity[peaks])
        except TypeError as e:
            print(f"Error: {e}, s={s}")

    spp, sdd = np.hstack(sp), np.hstack(sd)
    return spp, sdd

def histogram_peak_extraction(
        spp : FARRAY_1D,
        sdd : FARRAY_1D,
        sf : float
        ) -> float:
    """
    Select the highest peak from the overall distribution of peaks found using
    the algorithm described in the documentation for the
    noise_robust_spline_peak_extraction function.

    There is a small amount of procesesing done to spp before the peak is
    calculated. Specifically, all peaks with a color greater than the median
    color - sf * the standard deviation of the color are rejected. Moreover,
    all peaks with a color less than the median color + sf * the standard
    deviation of the color are also rejected. Finally any peaks below a density
    of 0.2 are rejected. This final parameter may need to be tweaked and should
    change to a function argument in future.

    Once these cuts are made that this function simply selects the max values
    from the histogram of spp using numpy's auto binning algorithm. In future a
    more robust peak detection tequnique may be implimented.

    Parameters
    ----------
        spp : ndarray[float64]
            1D array of all the colors of all peaks extracted from every spline
            fit to the data
        sdd : ndarray[float64]
            1D array of the densities (heights) of each peak extracted from
            every spline fit to the data
        sf : float
            Sigma distance to cut color based on

    Returns
    -------
        cHighest : float
            color of highest peak in the spline peak distribution.
            If spp has a length of 0 (i.e. no point falls within the bin)
            then np.nan is returned.
    """
    if spp.shape[0] == 0:
        return np.nan
    medianColor = np.median(spp)
    stdColor = np.std(spp)

    conditional = (spp >= medianColor-sf*stdColor) & (spp < medianColor+sf*stdColor) & (sdd >= 0.20)

    n, b = np.histogram(spp[conditional], bins='auto', density=True)
    c = (b[1:]+b[:-1])/2

    cHighest = c[n.argsort()[::-1]][0]

    return cHighest

def percentage_within_n_standard_deviations(n):
    cdf_n = norm.cdf(n)
    percentage = (2 * cdf_n - 1) * 100
    return percentage

def spline_based_density_peak_extraction(
        color : FARRAY_1D,
        density : FARRAY_1D,
        smin : float,
        smax : float,
        sn : int,
        sf : float
        ) -> Tuple[float, float, float]:
    """
    Extract the peak density of a density vs. color distribution using a spline
    based ensembell smoothing technique. Each spline will have a different
    smoothing factor s which are iterated over and defined in np.linspace(smin,
    smax, sn). The principal with this is that the higher the smoothing factor
    the more noise will be smeared over. However, too high and true peaks will
    be smoothed over.

    Given that we do not have a priori knowledge of which smoothing factor is
    ideal for a given density profile / population we would like to avoid
    selecting a fixed smoothing factor. Instead we could exploit the property
    that peaks from noise should smooth out before true peaks.

    What we could do is fit sn splines each with a slightly larger smoothing
    factor than the pervious spline. For each spline we then use a peak
    detection algorithm to identify peaks and store all of them in an array.

    True peaks should be detected more often than peaks from noise therefore
    in true peaks should dominate in the distribution of peak colors.

    Parameters
    ----------
        color : ndarray[float64]
            Color of each target in a CMD of shape m.
        density : ndarray[float64]
            Density at each point in a CMD of shape m.
        smin : float
            Miniumum smoothing factor to use. Generally this should be small
            but non zero (0.1 has worked well for me)
        smax : float
            Maximum smoothing factor to use. I find that its not good to push
            this much past 1
        sn : int
            Overall number of splines to fit. The spline fitting routines in
            scipy, which this function makes use of, are well optimized so
            you can push this pretty high without too many preformance issues.
            I find that a value of 100-200 works well. However, this will depend
            on your specific density profile.
        sf : float
            Sigma distance to cut color based on

    Returns
    -------
        cHighest : float
            color of highest peak in the spline peak distribution. In the event
            that the spline parameter space is extremply smooth and only one
            peak is extracted over all smoothing factors then this is simply
            the value of that peak.
        color5th : float
            5th percentile of the color distribution. This is used to set the
            lower bound of the color range for the fiducial line.
        color95th : float
            95th percentile of the color distribution. This is used to set the
            upper bound of the color range for the fiducial line.
    """
    spp, sdd = noise_robust_spline_peak_extraction(color, density, smin, smax, sn)

    if np.unique(spp).shape[0] == np.unique(sdd).shape[0] != 1:
        cHighest = histogram_peak_extraction(spp, sdd, sf)
    else:
        cHighest = np.unique(spp)[0]
    cHighest = color[np.argmax(density)]
    sigma = 1
    percentile = 100 - percentage_within_n_standard_deviations(sigma)
    densityPercentile = np.percentile(density, percentile)
    shiftedDensity = density - densityPercentile
    roots = list()
    for idx, (ld, ud, lc, uc) in enumerate(zip(shiftedDensity[:-1], shiftedDensity[1:], color[:-1], color[1:])):
        if np.sign(ld) != np.sign(ud):
            m = (ld - ud) / (lc - uc)
            roots.append( lc - (ld / m) )
    if len(roots) == 0 or len(roots) == 1:
        print("Not enough roots found")
        lowerBoundIntersection = np.nan
        upperBoundIntersection = np.nan
    elif len(roots) == 2:
        lowerBoundIntersection = roots[0]
        upperBoundIntersection = roots[1]
    else:
        lowerBoundIntersection = np.min(roots)
        upperBoundIntersection = np.max(roots)
    return cHighest, lowerBoundIntersection, upperBoundIntersection

def approximate_fiducial_line_function(
        color : FARRAY_1D,
        mag : FARRAY_1D,
        percLow : float = 1,
        percHigh : float = 99,
        binSize : float = 0.1,
        allowMax : bool = False,
        ) -> Callable:
    """
    Get an approximate fiducua line from a CMD using the basic ridge bounding
    algorithm

    Parameters
    ----------
        color : ndarray[float64]
            Color of each target in a CMD of shape m.
        mag : ndarray[float64]
            Magnitude of each target in a CMD of shape m.
        percLow : float, default=1
            Lower bound percentile to base range on
        percHigh : float, default=99
            Upper bound percentile to base range on
        binSize : float
            Spacing between each left bin edge to each right bin edge
        allowMax : bool, default=False
            If true then the ridge bounding algorithm will allow the maximum
            value in the color distribution to be used as a fiducial point if
            the gaussian fitting fails. If false and if the gaussian curve fit
            failes then a nan will be used.

    Returns
    -------
        ff : Callable
            1d linear interpolation between the calculate fiducial points
            using the ridge bounding method. ff is parmeterized as a function
            of magnitude (so as to make it a one-to-one function)

    """
    binsLeft, _ = mag_bins(mag, percLow, percHigh, binSize)
    fiducialLine = ridge_bounding(color, mag, binsLeft, allowMax=allowMax)
    ff = interp1d(fiducialLine[1], fiducialLine[0], bounds_error=False, fill_value='extrapolate')
    return ff

def verticalize_CMD(
        color : FARRAY_1D,
        mag : FARRAY_1D,
        percLow : float = 1,
        percHigh : float = 99,
        binSize : float = 0.1,
        allowMax : bool = False,
        ) -> Tuple[FARRAY_1D, Callable]:
    """
    Given some CMD fit an approximate fiducual line and use that to verticalize
    the CMD along the main sequence and the RGB.

    Parameters
    ----------
        color : ndarray[float64]
            Color of each target in a CMD of shape m.
        mag : ndarray[float64]
            Magnitude of each target in a CMD of shape m.
        percLow : float, default=1
            Lower bound percentile to base range on
        percHigh : float, default=99
            Upper bound percentile to base range on
        binSize : float
            Spacing between each left bin edge to each right bin edge
        allowMax : bool, default=False
            If true then the ridge bounding algorithm will allow the maximum
            value in the color distribution to be used as a fiducial point if
            the gaussian fitting fails. If false and if the gaussian curve fit
            failes then a nan will be used.

    Returns
    -------
        vColor : ndarray[float64]
            Verticalized color, indexed the same as color and mag
        ff : Callable
            1d linear interpolation between the calculate fiducial points
            using the ridge bounding method. ff is parmeterized as a function
            of magnitude (so as to make it a one-to-one function)

    """
    ff = approximate_fiducial_line_function(color, mag, percLow=percLow, percHigh=percHigh, binSize=binSize, allowMax=allowMax)
    vColor = color - ff(mag)
    return vColor, ff

def perp_distance(p0, p1, theta):
    a = np.cos(theta)*(p1[1]-p0[1])
    b = np.sin(theta)*(p1[0]-p0[0])
    return np.abs(a-b)

def fiducial_line(
        filter1 : Union[FARRAY_1D, pd.Series],
        filter2 : Union[FARRAY_1D, pd.Series],
        error1 : Union[FARRAY_1D, pd.Series],
        error2 : Union[FARRAY_1D, pd.Series],
        reverseFilterOrder : bool = False,
        mcruns : int = 10,
        convexHullPoints : int = 100,
        binSize : float = 0.1,
        percLow : float = 1,
        percHigh : float = 99,
        appxBinSize : float = 0.1,
        splineSmoothMin : float = 0.1,
        splineSmoothMax : float = 1,
        splineSmoothN : int = 200,
        colorSigCut : float = 5,
        pbar : bool = True,
        allowMax : bool = False,
        verbose : bool = False,
        cacheDensity : bool = False,
        cacheDensityName : str = 'CMDDensity.npz',
        minMagCut : float = -np.inf,
        uni_density: bool = False,
        ) -> FARRAY_2D_2C:
    """


    Parameters
    ----------
        filter1 : Union[ndarray[float64], pd.Series]
            First filter, will be A in A-B color
        filter2 : Union[ndarray[float64], pd.Series]
            Second filter, will be B in A-B color
        error1 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter1.
        error2 : Union[ndarray[float64], pd.Series]
            One sigma uncertainties in Photometry from filter2.
        reverseFilterOrder : bool, default=False
            Flag to determine which filter is used as the magnitude. If
            reverseFilterOrder is false then filter1 is is the magnitude. If
            reverseFilterOrder is True then filter2 is the magnitude
        mcruns : int, default=10
            Number of monte carlo runs to use when calculating the density. Note
            that increasing this will linearlly slow down your code. If mcruns
            is set to 1 then the density will be calculated using the
            nominal values (no monte carlo).
        convexHullPoints : int, default=100
            Number of closest points to considered when finding the convex hull
            used to define the area in the instantaious_hull_density function.
        binSize : int, default=0.1
            Bin size to use when applying the spline smoothing algorithm
        percLow : float, default=1
            Overall lower percentile bound to cut magnitude on.
        percHigh : float, default=99
            Overall upper percentile bound to cut magnitude on.
        splineSmoothMin : float, default=0.1
            Miniumum smoothing factor to use. Generally this should be small
            but non zero (0.1 has worked well for me)
        splineSmoothMax : float, default=1
            Maximum smoothing factor to use. I find that its not good to push
            this much past 1
        splineSmoothN : int, default=200
            Overall number of splines to fit. The spline fitting routines in
            scipy, which this function makes use of, are well optimized so
            you can push this pretty high without too many preformance issues.
            I find that a value of 100-200 works well. However, this will depend
            on your specific density profile.
        colorSigCut : float, default=5
            Sigma distance to cut color based on
        pbar : bool, default=True
            Flag controlling whether a progress bar is written to standard output.
            This will marginally slow down your code; however, its a very small
            effect and I generally find it helpful to have this turned on.
        allowMax : bool, default=False
            If true then the ridge bounding algorithm will allow the maximum
            value in the color distribution to be used as a fiducial point if
            the gaussian fitting fails. If false and if the gaussian curve fit
            failes then a nan will be used.
        verbose : bool, default=False
            Flag to control whether or not to print out information about the
            fiducial line fitting process. This is useful for debugging.
        cacheDensity : bool, default=False
            Flag to control whether or not to cache the density calculation.
            This can be useful if you are going to be calling this function
            multiple times with the same data. However, if you are going to be
            calling this function multiple times with different data then
            you should not use this flag.
        cacheDensityName : str, default='CMDDensity.npz'
            Name of the file to use to cache the density calculation. This is
            only used if cacheDensity is True. If the file does not
            exist then it will be created. If the file does exist then it will
            be loaded.
        minMagCut : float, default=-np.inf
            Minimum magnitude to cut on. This is useful if you want to cut out
            the RGB. Note that this is applied before the overall magnitude
            cut (percLow and percHigh) is applied.

    Returns
    -------
        fiducial : ndarray[[ndarray[float64], ndarray[float64]]]
            Ridge line fit to a1 and a2. ridgeLine[0, :] gives the coordinates
            of the ridge line along the a1 axis, while ridgeLine[1, :] gives
            the coordinates of the ridgeLine along the a2 axis (these
            are the infered centers points of the bins)
    """
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if verbose:
        logger.setLevel(logging.INFO)
        handler.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        handler.setLevel(logging.WARNING)


    warnings.showwarning = warning_traceback
    color, mag = color_mag_from_filters(filter1, filter2, reverseFilterOrder)

    binsLeft, binsRight = mag_bins(mag, percLow, percHigh, binSize)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1,2)
    #
    vColor, ff = verticalize_CMD(color, mag, percLow, percHigh, appxBinSize, allowMax=allowMax)
    # axs[0].scatter(vColor, mag, s=1)
    #
    # axs[1].scatter(color, mag, s=1)
    # axs[0].invert_yaxis()
    # axs[1].invert_yaxis()
    # plt.show()

    if cacheDensity and os.path.exists(cacheDensityName):
        logger.info("Using cached density...")
        loaded = np.load(cacheDensityName)
        density = loaded["density"]
        mcrunsLoaded = loaded["mcruns"]
        if mcrunsLoaded != mcruns:
            logger.debug(f"mcrunsLoaded: {mcrunsLoaded}")
            raise ValueError(f"mcruns does not match cached value! ({mcruns} != {mcrunsLoaded})")
    else:
        density = MC_convex_hull_density_approximation(
                filter1,
                filter2,
                error1,
                error2,
                reverseFilterOrder,
                convexHullPoints=convexHullPoints,
                mcruns=mcruns,
                pbar=pbar)

        if cacheDensity:
            with open(cacheDensityName, 'wb') as f:
                np.savez(f , density=density, mcruns=mcruns)

    fiducial=np.zeros(shape=(binsLeft.shape[0],5))
    logger.info("Fitting fiducial line to density...")
    for binID, (left, right) in tqdm(enumerate(zip(binsLeft, binsRight)), disable=verbose, desc="Fitting fiducial line to density", total=len(binsLeft)):
        logger.info(f"Working on bin {binID}")
        logger.info(f"\tDensity...")
        logger.info(f"\tSpline based density peak extraction...")
        dC, cC = density_color_cut(density, vColor, mag, left, right)
        cHighest, c5, c95 = spline_based_density_peak_extraction(
                cC,
                dC,
                splineSmoothMin,
                splineSmoothMax,
                splineSmoothN,
                colorSigCut
                )
        logger.info(f"\tFiducial point found at {cHighest}")
        m = (left + right)/2
        fiducial[binID,0] = cHighest
        fiducial[binID,1] = m
        fiducial[binID,2] = c5
        fiducial[binID,3] = c95

    logger.info("Completed fitting fiducial line to density!")

    fiducial[:,0] = fiducial[:,0] + ff(fiducial[:, 1])
    fiducial[:,2] = fiducial[:,2] + ff(fiducial[:, 1])
    fiducial[:,3] = fiducial[:,3] + ff(fiducial[:, 1])

    slope = np.gradient(fiducial[:,3], fiducial[:,1])
    theta = np.arctan(slope)
    fiducial[:,4] = perp_distance(fiducial[:,2], fiducial[:,3], theta)

    # remove all rows of fiducial which contain a nan value
    logger.info("Removing nan values from fiducial line...")
    fiducial = fiducial[~np.isnan(fiducial).any(axis=1)]

    # fiducial = fiducial[fiducial[:,1] > minMagCut]

    return fiducial
