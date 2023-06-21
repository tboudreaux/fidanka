from fidanka.fiducial.utils import percentile_range, median_ridge_line_estimate
import numpy as np
import numpy.typing as npt

import emcee
from multiprocessing import Pool

IARRAY_1D = npt.NDArray[np.int32]

FARRAY_1D = npt.NDArray[np.float64]
FARRAY_2D_2C = npt.NDArray[[FARRAY_1D, FARRAY_1D]]
FARRAY_2D_3C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D]]
FARRAY_2D_4C = npt.NDArray[[FARRAY_1D, FARRAY_1D, FARRAY_1D, FARRAY_1D]]

R2_VECTOR = npt.NDArray[[np.float64, np.float64]]


def plm(color, mag, piecewise_linear, binsLeft, binsRight, i):
    fiducial = np.zeros(shape=(binsLeft.shape[0], 5))
    masks = [
        ((mag >= binsLeft[i]) & (mag < binsRight[i])) for i in range(len(binsLeft))
    ]
    binned_mag = [mag[mask] for mask in masks]
    binned_color = [color[mask] for mask in masks]
    colorLeft = [min(binned_color[i]) for i in range(len(binned_color))]
    colorRight = [max(binned_color[i]) for i in range(len(binned_color))]
    color_error = np.sqrt(error1**2 + error2**2)
    binned_color_error = [color_error[mask] for mask in masks]
    cbin, m, _, _ = median_ridge_line_estimate(
        color, mag, binsLeft, binsRight, allowMax=allowMax
    )
    nwalkers = piecewise_linear[0]
    theta = np.concatenate((cbin, m))
    theta = np.tile(theta, (nwalkers, 1))
    stds = np.concatenate(
        (
            np.array([np.std(binned_color[i]) for i in range(len(cbin))]),
            np.array([np.std(binned_mag[i]) for i in range(len(cbin))]),
        )
    )
    for i in range(nwalkers):
        if i != 0:
            theta[i] += np.random.normal(0, stds, len(cbin) * 2)
    nwalkers, ndim = theta.shape
    print("Number of dimension = {}".format(ndim))
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_probability,
            args=(
                binned_mag,
                binned_color,
                binned_color_error,
                binsLeft,
                binsRight,
                colorLeft,
                colorRight,
            ),
            pool=pool,
        )
        sampler.run_mcmc(
            theta, piecewise_linear[1], progress=True, skip_initial_state_check=True
        )
    flat_samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    print("Log_probability of initial guess is {}".format(log_probs[0]))
    print("Log_probability of best match is {}".format(np.max(log_probs)))
    best_fit_val = flat_samples[np.argmax(log_probs)]
    for binID in range(len(binsLeft)):
        c5, c95 = percentile_range(binned_color[binID], 5, 95)
        m = best_fit_val[binID + len(cbin)]
        cHighest = best_fit_val[binID]
        fiducial[binID, 0] = cHighest
        fiducial[binID, 1] = m
        fiducial[binID, 2] = c5
        fiducial[binID, 3] = c95
    return fiducial


def log_prior(
    theta: FARRAY_1D,
    binsLeft: FARRAY_1D,
    binsRight: FARRAY_1D,
    colorLeft: FARRAY_1D,
    colorRight: FARRAY_1D,
) -> float:
    """
    provide an uninformative prior for mcmc

    Parameters
    ----------
        theta: FARRAY_1D
            Parameters corresponding to the color (first half) and magnitude (
            second half) of each bin
        binsLeft : ndarray[float64]
            left edges of bins in magnitude space
        binsRight : ndarray[float64]
            right edges of bins in magnitude space
        colorLeft : ndarray[float64]
            left edges of bins in color space
        colorRight : ndarray[float64]
            right edges of bins in color space

    Returns
    -------
        prior:float
            a uniform prior within the boundary
    """
    cbin = theta[: int(len(theta) / 2)]
    m = theta[int(len(theta) / 2) :]
    conditions = [
        (
            (cbin[i] > colorLeft[i])
            & (cbin[i] < colorRight[i])
            & (m[i] > binsLeft[i])
            & (m[i] < binsRight[i])
        )
        for i in range(len(m))
    ]
    for condition in conditions:
        if not condition:
            return -np.inf
    return 0.0


def log_likelihood(
    theta: FARRAY_1D,
    binned_mag: FARRAY_2D_2C,
    binned_color: FARRAY_2D_2C,
    binned_color_err: FARRAY_2D_2C,
) -> float:
    """
    Calculate the logrithmic likelihood for given parameters

    Parameters
    ----------
        theta: FARRAY_1D
            Parameters corresponding to the color (first half) and magnitude (
            second half) of each bin
        binned_mag : FARRAY_2D_2C,
            binned magnitude of all input data
        binned_color
            binned color of all input data
        binned_color_err : FARRAY_2D_2C,
            binned color of all input data

    Returns
    -------
        log_like: float
            logrithmic likelihood of given set of parameters
    """

    cbin = theta[: int(len(theta) / 2)]
    m = theta[int(len(theta) / 2) :]
    ff = interp1d(m, cbin, bounds_error=False, fill_value="extrapolate")
    Num_bin = len(cbin)
    log_like = 0
    for i in range(Num_bin):
        if i == 0:
            model = ff(binned_mag[i])
            log_like += -0.5 * np.sum(
                (binned_color[i] - model) ** 2 / binned_color_err[i] ** 2
                + np.log(2 * np.pi * binned_color_err[i] ** 2)
            )
        elif i == Num_bin - 1:
            model = ff(binned_mag[i])
            log_like += -0.5 * np.sum(
                (binned_color[i] - model) ** 2 / binned_color_err[i] ** 2
                + np.log(2 * np.pi * binned_color_err[i] ** 2)
            )
        else:
            first_half_mask = binned_mag[i] <= m[i]
            second_half_mask = binned_mag[i] > m[i]
            model = ff(binned_mag[i][first_half_mask])
            log_like += -0.5 * np.sum(
                (binned_color[i][first_half_mask] - model) ** 2
                / binned_color_err[i][first_half_mask] ** 2
                + np.log(2 * np.pi * binned_color_err[i][first_half_mask] ** 2)
            )
            model = ff(binned_mag[i][second_half_mask])
            log_like += -0.5 * np.sum(
                (binned_color[i][second_half_mask] - model) ** 2
                / binned_color_err[i][second_half_mask] ** 2
                + np.log(2 * np.pi * binned_color_err[i][second_half_mask] ** 2)
            )
    return log_like


def log_probability(
    theta: FARRAY_1D,
    binned_color: FARRAY_2D_2C,
    binned_mag: FARRAY_2D_2C,
    binned_color_err: FARRAY_2D_2C,
    binsLeft: FARRAY_1D,
    binsRight: FARRAY_1D,
    colorLeft: FARRAY_1D,
    colorRight: FARRAY_1D,
) -> float:
    """
    Calculate the logrithmic probabilty for the parameter. Note, compare to
    the verticalized cmd method, this method also take into the consideration
    of the relation between neighbor bins.
    The main drawbacks of mcmc is that it is very ineffcient in recovering those
    parameters when they are highly correlated and the curse of dimensionality
    also suggests that using mcmc in this case can be very computationally
    expensive.

    Parameters
    ----------
        theta: FARRAY_1D,
            Parameters corresponding to the color (first half) and magnitude (
            second half) of each bin
        binned_mag : FARRAY_2D_2C,
            binned magnitude of all input data
        binned_color
            binned color of all input data
        binned_color_err : FARRAY_2D_2C,
            binned color of all input data
        binsLeft : ndarray[float64]
            left edges of bins in magnitude space
        binsRight : ndarray[float64]
            right edges of bins in magnitude space
        colorLeft : ndarray[float64]
            left edges of bins in color space
        colorRight : ndarray[float64]
            right edges of bins in color space

    Returns
    -------
        log_prop
            logrithmic probability of given sets of parameters
    """
    lp = log_prior(theta, binsLeft, binsRight, colorLeft, colorRight)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, binned_color, binned_mag, binned_color_err)
