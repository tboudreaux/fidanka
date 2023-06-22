from fidanka.isochrone.isochrone import interp_isochrone_age
from fidanka.misc.utils import pfD

import pytensor.tensor as pt
import pymc as pm
import pytensor
import numpy as np

from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
from fastdtw import fastdtw


class BolCorOp(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, isoDict, bc, filters, fiducial):
        self.isoDict = isoDict
        self.bc = bc
        self.filters = filters
        self.fiducial = fiducial.mean
        self.transposedFiducial = self.fiducial.T
        self.sigma = fiducial.confidence(0.68)
        print("BolCorOp init")

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        ((age, mu, E),) = inputs  # this will contain my variables

        # call the log-likelihood function
        isoAtAge = interp_isochrone_age(self.isoDict, age)
        isoFilter1Name = f"WFC3_UVIS_{self.filters[0]}_MAG"
        isoFilter2Name = f"WFC3_UVIS_{self.filters[1]}_MAG"
        isoFilter3Name = f"WFC3_UVIS_{self.filters[2]}_MAG"

        bolCorrected = self.bc.apparent_mags(
            10 ** isoAtAge["log_Teff"],
            isoAtAge["log_g"],
            isoAtAge["log_L"],
            Av=E,
            mu=mu,
        )
        isoColor = bolCorrected[isoFilter1Name] - bolCorrected[isoFilter2Name]
        isoMag = bolCorrected[isoFilter3Name]
        f = interp1d(isoMag, isoColor, bounds_error=False, fill_value="extrapolate")

        sortedFiducial = self.transposedFiducial[
            self.transposedFiducial[:, 1].argsort()
        ]
        minDist = np.empty(shape=(sortedFiducial.shape[0]))
        minDist[:] = np.nan

        # For each point in the fiducual line find the minimum possible distance
        # to the isochrone
        for idx, point in enumerate(sortedFiducial):
            d = pfD(point, f)
            # plt.plot([point[0], f(point[1])], [point[1], point[1]], 'k')
            nearestPoint = minimize(d, 0, method="Nelder-Mead")
            if not nearestPoint.success:
                print("FAIL")
            else:
                minDist[idx] = d(nearestPoint.x[0])

        target = np.array([sortedFiducial[:, 1], f(sortedFiducial[:, 1])]).T
        source = np.flip(sortedFiducial, axis=1)
        dtwDist = fastdtw(source, target, dist=euclidean)
        minDist = minDist[~np.isnan(minDist)]
        chi2 = np.sum(np.apply_along_axis(lambda x: x**2, 0, minDist))
        preDTWchi2 = chi2
        chi2 += np.sqrt(dtwDist[0])
        chi2nu = chi2 / (sortedFiducial.shape[0] + 1)

        outputs[0][0] = np.array(chi2nu)  # output the log-likelihood
