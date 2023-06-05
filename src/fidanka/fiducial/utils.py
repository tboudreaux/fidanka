import numpy as np
import numpy.typing as npt

from sklearn.mixture import GaussianMixture

from typing import List, Tuple, Union

IARRAY_1D = npt.NDArray[np.int32]

FARRAY_1D = npt.NDArray[np.float64]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]
FARRAY_2D_3C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D]]
FARRAY_2D_4C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D, FARRAY_1D]]

R2_VECTOR = npt.NDArray[[np.float64, np.float64]]


def clean_bins(
        colorBins : List[FARRAY_1D],
        magBins : List[FARRAY_1D],
        densityBins : List[FARRAY_1D],
        sigma : float = 5,
        iterations : int = 1
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
        cut = (color >= meanColor - sigma*stdColor) & (color < meanColor + sigma*stdColor)
        newColorBins.append(color[cut])
        newMagBins.append(mag[cut])
        newDensityBins.append(density[cut])

    # Repeat the process if needed
    if iterations > 1:
        return clean_bins(newColorBins, newMagBins, newDensityBins, sigma=sigma, iterations=iterations-1)

    return newColorBins, newMagBins, newDensityBins

def normalize_density_magBin(
        color : FARRAY_1D,
        mag : FARRAY_1D,
        density : FARRAY_1D,
        binSize : float = 0.1
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

    Returns
    -------
        normDensity : FARRAY_1D
            Normalized density of stars in the CMDs.

    """
    normDensity = np.zeros(shape=color.shape[0])
    for IDx, (c, m, d) in tqdm(enumerate(zip(color, mag, density)), total=len(density)):
        cut = (mag > m-binSize/2) & (mag <= m+binSize/2)
        binDensity = density[cut]
        meanBinDensity = np.mean(binDensity)
        normalizedDensity = d/meanBinDensity
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
