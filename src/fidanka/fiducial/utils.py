import numpy as np
import pandas as pd
import numpy.typing as npt

from sklearn.mixture import GaussianMixture

from tqdm import tqdm

from typing import List, Tuple, Union

IARRAY_1D = npt.NDArray[np.int32]

FARRAY_1D = npt.NDArray[np.float64]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]
FARRAY_2D_3C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D]]
FARRAY_2D_4C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D, FARRAY_1D]]

R2_VECTOR = npt.NDArray[[np.float64, np.float64]]


def clean_bins(
    colorBins: List[FARRAY_1D],
    magBins: List[FARRAY_1D],
    densityBins: List[FARRAY_1D],
    sigma: float = 5,
    iterations: int = 1,
) -> Tuple[List[FARRAY_1D], List[FARRAY_1D], List[FARRAY_1D]]:
    """
    Remove high sigma outliers from the bins. Repeat the process iterativly.

    Parameters
    ----------
        colorBins : List[FARRAY_1D]
            List of color bins.
        magBins : List[FARRAY_1D]
            List of magnitude bins.
        densityBins : List[FARRAY_1D]
            List of density bins.
        sigma : float, default=5
            Sigma value to use for the outlier removal.
        iterations : int, default=1
            Number of iterations to perform.

    Returns
    -------
        Tuple[List[FARRAY_1D], List[FARRAY_1D], List[FARRAY_1D]]
            Cleaned color, magnitude and density bins.

    Examples
    --------
    >>> colorBins = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    >>> magBins = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    >>> densityBins = [np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5])]
    >>> clean_bins(colorBins, magBins, densityBins, sigma=1, iterations=1)

    """
    assert iterations > 0, "iterations must be greater than 0"

    newColorBins = list()
    newMagBins = list()
    newDensityBins = list()
    for i, (color, mag, density) in enumerate(zip(colorBins, magBins, densityBins)):
        meanColor = np.mean(color)
        stdColor = np.std(color)
        cut = (color >= meanColor - sigma * stdColor) & (
            color < meanColor + sigma * stdColor
        )
        newColorBins.append(color[cut])
        newMagBins.append(mag[cut])
        newDensityBins.append(density[cut])

    # Repeat the process if needed
    if iterations > 1:
        return clean_bins(
            newColorBins,
            newMagBins,
            newDensityBins,
            sigma=sigma,
            iterations=iterations - 1,
        )

    return newColorBins, newMagBins, newDensityBins


def normalize_density_magBin(
    color: FARRAY_1D,
    mag: FARRAY_1D,
    density: FARRAY_1D,
    binSize: float = 0.1,
    pbar: bool = False,
) -> FARRAY_1D:
    """
    Normalize the density of each point on a CMD by the mean of the densities
    within binSize of the point. This allows densities to be compared between
    the RGB and MS more easily.

    Parameters
    ----------
        color : FARRAY_1D
            Colors of stars in the CMD.
        mag : FARRAY_1D
            Magnitudes of stars in the CMD.
        density : FARRAY_1D
            Density of stars in the CMDs.
        binSize : float, default=0.1
            Size of the bins to use for the normalization.
        pbar : bool, default=False
            Show a progress bar.

    Returns
    -------
        normDensity : FARRAY_1D
            Normalized density of stars in the CMDs.

    """
    normDensity = np.zeros(shape=color.shape[0])
    for IDx, (c, m, d) in tqdm(
        enumerate(zip(color, mag, density)), total=len(density), disable=not pbar
    ):
        cut = (mag > m - binSize / 2) & (mag <= m + binSize / 2)
        binDensity = density[cut]
        meanBinDensity = np.mean(binDensity)
        normalizedDensity = d / meanBinDensity
        normDensity[IDx] = normalizedDensity
    return normDensity


def GMM_component_measurment(xB, yB, n=50):
    """
    Fit a Gaussian Mixture Model to each bin of a CMD and return the mean of
    each component.

    Parameters
    ----------
        xB : List[FARRAY_1D]
            List of color bins.
        yB : List[FARRAY_1D]
            List of magnitude bins.
        n : int, default=50
            Number of components to fit to each bin.

    Returns
    -------
        gmm_means : FARRAY_2D_2C
            List of the means of the Gaussian Mixture Models fit to each bin.

    """
    n_bins = len(xB)
    gmm_means = []

    for i_bin in range(n_bins):
        x_bin = np.expand_dims(xB[i_bin], axis=1)
        y_bin = yB[i_bin]

        # Normalize densities so they can be used as weights
        weights = y_bin / np.sum(y_bin)

        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(x_bin, weights)

        gmm_means.append(gmm.means_)

    gmm_means = np.squeeze(gmm_means)
    return gmm_means


def median_ridge_line_estimate(
    color,
    mag,
    density,
    binSize="uniformCS",
    max_Num_bin=False,
    binSize_min=0.1,
    sigmaCut=3,
    cleaningIterations=10,
    components=100,
    normBinSize=0.3,
    targetStat=250,
):
    colorBins, magBins, densityBins = bin_color_mag_density(
        color,
        mag,
        density,
        binSize=binSize,
        max_Num_bin=max_Num_bin,
        binSize_min=binSize_min,
        targetStat=targetStat,
    )
    colorBins, magBins, densityBins = clean_bins(
        colorBins, magBins, densityBins, sigma=sigmaCut, iterations=cleaningIterations
    )
    ridgeLine = np.zeros(shape=(4, len(colorBins)))

    components = min(100, int(len(colorBins) * 0.1))
    gmm_means = GMM_component_measurment(colorBins, densityBins, n=components)
    color = np.median(gmm_means, axis=1)
    mag = np.array([np.mean(x) for x in magBins])

    lowPercentile = np.percentile(gmm_means, 68.97, axis=1)
    highPercentile = np.percentile(gmm_means, 100 - 68.97, axis=1)

    ridgeLine[0] = color
    ridgeLine[1] = mag
    ridgeLine[2] = lowPercentile
    ridgeLine[3] = highPercentile

    return ridgeLine


def percentile_range(
    X: Union[FARRAY_1D, pd.Series], percLow: float, percHigh: float
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


def bin_color_mag_density(
    color: FARRAY_1D,
    mag: FARRAY_1D,
    density: FARRAY_1D,
    binSize: Union[str, float] = "uniformCS",
    percLow: Union[float, None] = None,
    percHigh: Union[float, None] = None,
    max_Num_bin: Union[bool, int] = False,
    binSize_min: float = 0.1,
    targetStat: int = 1000,
) -> Tuple[FARRAY_1D, FARRAY_1D, FARRAY_1D]:
    """
    Use the bin edges from mag_bins to split the color, mag, and density arrays
    into lists of arrays.

    Parameters
    ----------
        color : np.ndarray[float64]
            1D array of color of shape m for the stars in the CMD
        mag : np.ndarray[float64]
            1D array of magnitude of shape m for the stars in the CMD
        density : np.ndarray[float64]
            1D array of density of shape m for the stars in the CMD
        percLow : float
            Lower bound percentile to base range on
        percHigh : float
            Upper bound percentile to base range on
        binSize : Union[str, float]
            Size of the bins to use. If 'adaptive' will attempt to keep
            counting statistics the same throughout.
        max_Num_bin : Union[bool, int]
            False or `0` will not limit the number of bins. Any other integer
            will limit the number of bins to that number.
        binSize_min : float
            Minimum bin size to use when using adaptive binning
        targetStat : int, default=1000
            Target number of stars per bin when using uniformCS binning

    Returns
    -------
        colorBins : List[np.ndarray[float64]]
            List of arrays of color of shape m for the stars in the CMD
        magBins : List[np.ndarray[float64]]
            List of arrays of magnitude of shape m for the stars in the CMD
        densityBins : List[np.ndarray[float64]]
            List of arrays of density of shape m for the stars in the CMD
    """
    if max_Num_bin == 0:
        max_Num_bin = False
    left, right = mag_bins(
        mag, percHigh, percLow, binSize, binSizeMin=binSize_min, targetStat=targetStat
    )

    colorBins = list()
    magBins = list()
    densityBins = list()
    for l, r in zip(left, right):
        condition = (mag >= l) & (mag < r)
        colorBins.append(color[condition])
        magBins.append(mag[condition])
        densityBins.append(density[condition])
    return colorBins, magBins, densityBins


def mag_bins(
    mag: Union[FARRAY_1D, pd.Series],
    percHigh: Union[float, None],
    percLow: Union[float, None],
    binSize: Union[str, float],
    maxNumBins: Union[bool, int] = False,
    binSizeMin: float = 0.1,
    targetStat: float = 1000,
) -> Tuple[FARRAY_1D, FARRAY_1D]:
    """
    Find the left and right edges of bins in magnitude space between magnitudes
    percLow and percHigh percentiles with a bin size of binSize. Find suitable
    binsize if choose to use adapative bin size

    Parameters
    ----------
        mag : Union[ndarray[float64], pd.Series]
            1D array of magnitude of shape m
        percLow : float
            Lower bound percentile to base range on
        percHigh : float
            Upper bound percentile to base range on
        binSize : Union[str, float]
            Spacing between each left bin edge to each right bin edge. If
            'adaptive' will attempt to keep match the target number of bins
            over the range. If binSize is set to uniformCS then the bin size
            will be adjusted so that the counting statistics are roughly the
            same in each bin.
        maxNumBins: Union[str, int], default=False
            maximum number of bins
        binSizeMin : float, default=0.1
            Minimum bin size to use when using adaptive binning
        targetStat : int, default=100
            Target number of stars in each bin when using uniformCS binning

    Returns
    -------
        binsLeft : ndarray[float64]
            left edges of bins in magnitude space
        binsRight : ndarray[float64]
            right edges of bins in magnitude space
    """
    if binSize != "uniformCS":
        assert percLow is not None, "percLow must be set if binSize is not uniformCS"
        assert percHigh is not None, "percHigh must be set if binSize is not uniformCS"
    if binSize == "adaptive":
        lenMag = len(mag)
        if maxNumBins == False:
            binCountWanted = 50
        else:
            binCountWanted = np.max((int(np.ceil(lenMag // maxNumBins)), 50))

        magSort = np.sort(mag)
        binsLeft = [magSort[0]]
        binsRight = []
        i = 0
        iLeft = 0
        while i < lenMag:
            if magSort[i] - binsLeft[-1] < binSizeMin:
                i += 1
            else:
                if i - iLeft >= binCountWanted:
                    binsRight.append(magSort[i])
                    binsLeft.append(magSort[i])
                    iLeft = i
                i += 1
        binsLeft = np.array(binsLeft[:-1])
        binsRight = np.append(np.array(binsRight[:-1]), magSort[-1] + 0.001)
    if binSize == "uniformCS":
        srtedIDX = np.argsort(mag)
        sMag = mag[srtedIDX]

        # Type consistency
        if isinstance(sMag, pd.Series):
            sMag = sMag.values
        Num_star = len(sMag)
        binsLeft = [sMag[0]]
        binsRight = []
        idx_left = 0
        idx_right_count = idx_left + targetStat
        idx_right_mag = np.searchsorted(sMag, binsLeft[-1] + binSizeMin)
        idx_right_max = max(idx_right_count, idx_right_mag)
        while idx_right_max < Num_star:
            binsRight.append(sMag[idx_right_max])
            binsLeft.append(sMag[idx_right_max])
            idx_left = idx_right_max
            idx_right_count = idx_left + targetStat
            idx_right_mag = np.searchsorted(sMag, binsLeft[-1] + binSizeMin)
            idx_right_max = max(idx_right_count, idx_right_mag)
        binsLeft = np.array(binsLeft[:-1])
        binsRight[-1] = sMag[-1]
        binsRight = np.array(binsRight)

    else:
        magRange = percentile_range(mag, percLow, percHigh)
        binsLeft = np.arange(magRange[1], magRange[0], binSize)
        binsRight = binsLeft + binSize
    return binsLeft, binsRight
