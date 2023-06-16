import numpy as np
import numpy.typing as npt
from typing import Callable, Tuple, Union
from scipy.interpolate import interp1d
import logging

FARRAY_1D = npt.NDArray

def inverse_cdf_sample(
        f : Callable[[FARRAY_1D], FARRAY_1D],
        x : FARRAY_1D = None
        ) -> Callable[[FARRAY_1D], FARRAY_1D]:
    """
    Generate a function that samples from the inverse CDF of a given function.

    Parameters
    ----------
        f : Callable[[FARRAY_1D], FARRAY_1D]
            Function to sample from.
        x : FARRAY_1D, default=None
            Domain of the function. If None, defaults to np.linspace(0,1,100000).

    Returns
    -------
        inverse_cdf : Callable[[FARRAY_1D], FARRAY_1D]
            Function that samples from the inverse CDF of f. To evaluate the
            function, pass an array of uniform random numbers between 0 and 1.

    Examples
    --------
    Let's sample from the inverse CDF of a Gaussian distribution. First, we
    define the Gaussian distribution.

    >>> def gaussian(x, mu=0, sigma=1):
    ...     return np.exp(-(x-mu)**2/(2*sigma**2))

    Then, we generate the inverse CDF function.

    >>> inverse_cdf = inverse_cdf_sample(gaussian)

    Finally, we sample from the inverse CDF.

    >>> inverse_cdf(np.random.random(10))
    """
    if x is None:
        x = np.linspace(0,1,100000)

    y = f(x)
    cdf_y = np.cumsum(y)
    cdf_y_norm = cdf_y/cdf_y.max()

    inverse_cdf = interp1d(
            cdf_y_norm,
            x,
            bounds_error=False,
            fill_value='extrapolate'
            )

    return inverse_cdf

def get_samples(
        n : int,
        f : Callable[[FARRAY_1D], FARRAY_1D],
        domain : FARRAY_1D = None
        ) -> FARRAY_1D:
    """
    Sample n values from a given function. The function does not have to be
    a normalized PDF as the function will be normalized before sampling.

    Parameters
    ----------
        n : int
            Number of samples to draw.
        f : Callable[[FARRAY_1D], FARRAY_1D]
            Function to sample from.
        domain : FARRAY_1D, default=None
            Domain of the function. If None, defaults to np.linspace(0,1,100000).

    Returns
    -------
        samples : NDArray[float]
            Array of samples.

    Examples
    --------
    Let's sample 10 values from a quadratic function over the domain 0,2.

    >>> def quadratic(x):
    ...     return x**2

    >>> get_samples(10, quadratic, domain=np.linspace(0,2,1000))
    """

    uniformSamples = np.random.random(n)
    shiftedSamples = inverse_cdf_sample(f, x=domain)(uniformSamples)
    return shiftedSamples

def closest(
        array : FARRAY_1D,
        target : float
        ) -> Tuple[Union[FARRAY_1D, None], Union[FARRAY_1D, None]]:
    """
    Find the closest values above and below a given target in an array.
    If the target is in the array, the function returns the exact target value
    in both elements of the tuple. If the target is not exactly in the array,
    the function returns the closest value below the target in the first
    element of the tuple and the closest value above the target in the second
    element of the tuple. If the taret is below the minimum value in the array,
    the first element of the tuple is None. If the target is above the maximum
    value in the array, the second element of the tuple is None.

    Parameters
    ----------
        array : NDArray[float]
            Array to search.
        target : float
            Target value.

    Returns
    -------
        closest_lower : Union[NDArray[float], None]
            Closest value below the target. If the target is below the minimum
            value in the array, returns None.
        closest_upper : Union[NDArray[float], None]
            Closest value above the target. If the target is above the maximum
            value in the array, returns None.

    Examples
    --------
    Let's find the closest values above and below 5 in an array.

    >>> array = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> closest(array, 5)
    (5, 6)
    """
    exact_value = array[array == target]

    if exact_value.size > 0:
        return exact_value[0], exact_value[0]

    younger_ages = array[array < target]
    older_ages = array[array > target]

    if younger_ages.size == 0:
        closest_lower = None
    else:
        closest_lower = younger_ages[np.argmin(np.abs(younger_ages - target))]

    if older_ages.size == 0:
        closest_upper = None
    else:
        closest_upper = older_ages[np.argmin(np.abs(older_ages - target))]

    return closest_lower, closest_upper

def interpolate_arrays(
        array_lower : npt.NDArray,
        array_upper : npt.NDArray,
        target : float,
        lower : float,
        upper : float,
        joinCol : Union[int, None] = None
        ) -> npt.NDArray:
    """
    Interpolate between two arrays. The arrays must have the same shape.

    Parameters
    ----------
        array_lower : NDArray[float]
            Lower bounding array.
        array_upper : NDArray[float]
            Upper bounding array.
        target : float
            Target value to interpolate to.
        lower : float
            value at lower bounding array
        upper : float
            value at upper bounding array
        joinCol : int, default=None
            Column to join on. If None, assumes the arrays are parallel
    Returns
    -------
        interpolated_array : NDArray[float]
            Interpolated array at target value.

    Examples
    --------
    Let's interpolate between two arrays.

    >>> array_lower = np.array([1,2,3,4,5,6,7,8,9,10])
    >>> array_upper = np.array([11,12,13,14,15,16,17,18,19,20])

    >>> interpolate_arrays(array_lower, array_upper, 5.5, 5, 6)
    """
    if array_lower is None or array_upper is None:
        raise ValueError("Both arrays must be non-None")

    if not isinstance(array_lower, np.ndarray):
        array_lower = np.array(array_lower)

    if not isinstance(array_upper, np.ndarray):
        array_upper = np.array(array_upper)

    if joinCol is not None:
        shared = np.intersect1d(array_lower[:,joinCol], array_upper[:,joinCol])
        lowerMask = np.isin(array_lower[:,joinCol], shared)
        upperMask = np.isin(array_upper[:,joinCol], shared)
        array_lower = array_lower[lowerMask]
        array_upper = array_upper[upperMask]
    # Ensure both arrays have the same shape
    assert array_lower.shape == array_upper.shape, "Arrays must have the same shape"

    # Calculate the interpolation weights
    lower_weight = (upper - target) / (upper - lower)
    upper_weight = (target - lower) / (upper - lower)

    # Perform element-wise interpolation
    interpolated_array = (array_lower * lower_weight) + (array_upper * upper_weight)

    return interpolated_array

def get_logger(name, level=logging.INFO, flevel=logging.INFO, clevel=logging.WARNING):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        # create a file handler
        file_handler = logging.FileHandler('fidanka.log')
        file_handler.setLevel(flevel)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(clevel)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

