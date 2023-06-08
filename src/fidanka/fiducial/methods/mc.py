from fidanka.fiducial.utils import percentile_range, median_ridge_line_estimate

import emcee
from multiprocessing import Pool

def plm(color, mag, piecewise_linear, binsLeft, binsRight, i):
    fiducial=np.zeros(shape=(binsLeft.shape[0],5))
    masks = [((mag >= binsLeft[i]) & (mag < binsRight[i])) for i in range(len(binsLeft))]
    binned_mag = [mag[mask] for mask in masks]
    binned_color = [color[mask] for mask in masks]
    colorLeft = [min(binned_color[i]) for i in range(len(binned_color))]
    colorRight = [max(binned_color[i]) for i in range(len(binned_color))]
    color_error = np.sqrt(error1**2 + error2**2)
    binned_color_error = [color_error[mask] for mask in masks]
    cbin, m, _, _ = median_ridge_line_estimate(color, mag, binsLeft, binsRight, allowMax=allowMax)
    nwalkers = piecewise_linear[0]
    theta = np.concatenate((cbin,m))
    theta = np.tile(theta,(nwalkers,1))
    stds = np.concatenate((np.array([np.std(binned_color[i]) for i in range(len(cbin))]),np.array([np.std(binned_mag[i]) for i in range(len(cbin))])))
    for i in range(nwalkers):
        if i != 0:
            theta[i] += np.random.normal(0, stds, len(cbin)*2)
    nwalkers, ndim = theta.shape
    print("Number of dimension = {}".format(ndim))
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(binned_mag, binned_color, binned_color_error,binsLeft, binsRight, colorLeft, colorRight), pool=pool
        )
        sampler.run_mcmc(theta, piecewise_linear[1], progress=True, skip_initial_state_check = True)
    flat_samples = sampler.get_chain(flat=True)
    log_probs = sampler.get_log_prob(flat=True)
    print("Log_probability of initial guess is {}".format(log_probs[0]))
    print("Log_probability of best match is {}".format(np.max(log_probs)))
    best_fit_val = flat_samples[np.argmax(log_probs)]
    for binID in range(len(binsLeft)):
        c5, c95 = percentile_range(binned_color[binID],5,95)
        m = best_fit_val[binID + len(cbin)]
        cHighest = best_fit_val[binID]
        fiducial[binID,0] = cHighest 
        fiducial[binID,1] = m
        fiducial[binID,2] = c5
        fiducial[binID,3] = c95
    return fiducial
