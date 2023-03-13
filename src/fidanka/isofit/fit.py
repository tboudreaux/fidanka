from fidanka.isochrone.isochrone import shift_isochrone
from fidanka.isochrone.isochrone import interp_isochrone_age

import numpy as np
import numpy.typing as npt
from typing import Callable

from scipy.interpolate import interp1d
from scipy.optimize import minimize

R2_VECTOR = npt.NDArray[[np.float64, np.float64]]

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

def get_ISO_CMD_Chi2(iso, fiducialLine, filters=("F606W", "F814W", "F606W"), distance=0, Av=0, plot=False, ANIM=None):
    isoFilter1Name = f"WFC3_UVIS_{filters[0]}_MAG"
    isoFilter2Name = f"WFC3_UVIS_{filters[1]}_MAG"
    isoFilter3Name = f"WFC3_UVIS_{filters[2]}_MAG"

    isoColor = iso[isoFilter1Name] - iso[isoFilter2Name]
    isoMag = iso[isoFilter3Name]


    isoAptMag, isoAptColor = shift_isochrone(isoColor, isoMag, distance, Av)

    f = interp1d(isoAptMag, isoAptColor, bounds_error=False, fill_value='extrapolate')
    diffs = fiducialLine[:, 0] - f(fiducialLine[:, 1])
    sortedFiducial = fiducialLine[fiducialLine[:,1].argsort()]
    minDist = np.empty(shape=(sortedFiducial.shape[0]))
    minDist[:] = np.nan
    for idx, (point, d0) in enumerate(zip(sortedFiducial, diffs)):
        d = pfD(point, f)
        nearestPoint = minimize(d, 0, method='Nelder-Mead')
        if not nearestPoint.success:
            print("FAIL")
        else:
            minDist[idx] = d(nearestPoint.x[0])

    minDist = minDist[~np.isnan(minDist)]
    chi2 = np.sum(np.apply_along_axis(lambda x: x**2,0,minDist))/minDist.shape[0]
    return chi2

def optimize(photometry, fiducial, isochrone, filters,plot=False):
    age_d_E_opt = lambda iso, cmd, age, d, E: get_ISO_CMD_Chi2(
            interp_isochrone_age(iso, age),
            fiducial,
            distance=d,
            Av=E,
            filters=filters,
            plot=plot)

    optimized = minimize(
            lambda x: age_d_E_opt(isochrone, photometry, x[0], x[1], x[2]),
            [12, 10000, 0.06],
            bounds=[
                (5,20),
                (5000, None),
                (0,0.3)
                ],
            )
    return {'opt': optimized, 'iso': isochrone, 'fiducial': fiducial}

def order_best_fit_result(optimizationResults):
    comparison = {'A': list(), 'E': list()}
    for pop, popD in optimizationResults.items():
        for Y, YD in popD.items():
            for a, aD in YD.items():
                comparison[pop].append((aD['opt']['fun'], aD['opt']['x'], pop, Y, a))
        comparison[pop] = sorted(comparison[pop], key=lambda x: x[0])
    return comparison
