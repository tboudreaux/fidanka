from fidanka.isochrone.isochrone import shift_full_iso
from fidanka.population.utils import sample_n_masses
from fidanka.bolometric import BolometricCorrector
from fidanka.misc.utils import closest, interpolate_arrays, get_samples, get_logger
from fidanka.isochrone.MIST import read_iso, read_iso_metadata
from fidanka.population.artificialStar import artificialStar
from fidanka.population.ager import populationAgeDistribution
from fidanka.misc.utils import interpolate_keyed_arrays


import pickle as pkl
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import numpy.typing as npt

import warnings

from scipy.interpolate import interp1d, LinearNDInterpolator
from collections.abc import Iterable

from typing import Union, Tuple, List, Dict, Callable, Optional
from collections.abc import Sequence

FILTERPATTERN = re.compile(r"(?:(?:ACS_WFC_(F\d{3}W)_MAG)|(F\d{3}W))")
FARRAY_1D = npt.NDArray[np.float64]

# TODO: Updated module to use logger not print
# TODO: Make the artificial star test suite more complete
#       Impliment artstar test class which can take multiple
#       generators. Then the population object just takes
#       that object.


def mass_sample(
    n: int, mrange: Tuple[float, float] = (0.1, 1), alpha: float = -2.68
) -> FARRAY_1D:
    """
    Sample masses from a power law IMF.

    Parameters
    ----------
        n : int
            Number of samples to draw.
        mrange : Tuple[float, float], default=(0.1,1)
            Range of masses to sample from.
        alpha : float, default=-2.68
            Power law index of the IMF.

    Returns
    -------
        sample : NDArray[float]
            Array of sampled masses.

    Examples
    --------
    Let's sample 10 masses from a power law IMF with a power law index of -2.68
    between 0.1 and 1 solar masses.

    >>> mass_sample(10, mrange=(0.1,1), alpha=-2.68)
    """
    sample = np.zeros(n)
    m1 = mrange[1] ** (alpha + 1) - mrange[0] ** (alpha + 1)
    m2 = mrange[0] ** (alpha + 1)
    powmass = 1 / (alpha + 1)
    for i in range(n):
        xx = np.random.uniform(low=0, high=1)
        sm = (m1 * xx + m2) ** powmass
        sample[i] = sm
    return sample


def interpolate_eep_arrays(
    arr1: Sequence, arr2: Sequence, target: float, lower: float, upper: float
) -> Sequence:
    # TODO: Figure out if this function is still needed or if it has been superseeded by the interpolate
    # array function

    # Ensure arrays are numpy arrays
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    # Extract the EEP values from both arrays
    eep_arr1 = arr1[:, 0]
    eep_arr2 = arr2[:, 0]

    # Find the intersection of the EEP values in both arrays
    common_eeps = np.intersect1d(eep_arr1, eep_arr2)

    # Filter the arrays to keep only rows with common EEP values
    arr1_filtered = arr1[np.isin(eep_arr1, common_eeps)]
    arr2_filtered = arr2[np.isin(eep_arr2, common_eeps)]

    # Sort the filtered arrays by EEP values
    arr1_filtered = arr1_filtered[np.argsort(arr1_filtered[:, 0])]
    arr2_filtered = arr2_filtered[np.argsort(arr2_filtered[:, 0])]

    # Perform the linear interpolation element-wise
    interp_ratio = (target - lower) / (upper - lower)
    interpolated_arr = arr1_filtered + interp_ratio * (arr2_filtered - arr1_filtered)

    return interpolated_arr


def sum_mag(m1: float, m2: float) -> float:
    """
    Take the sum of two magnitudes.

    Parameters
    ----------
        m1 : float
            Magnitude of object 1
        m2 : float
            Magnitude of object 2

    Returns
    -------
        sumMagg : float
            Sum of magnitude m1 + m2 taken properly in log space.
    """
    return -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))


def sum_err_mag(m1: float, m2: float, s1: float, s2: float) -> float:
    """
    Find the sum of the errors of two magnitudes (general sum of uncertantiies
    in log_2.5 space)

    Parameters
    ----------
        m1 : float
            Magnitude of object 1
        m2 : float
            Magnitude of object 2
        s1 : float
            One sigma uncertantiies of magnitude of object 1
        s2 : float
            One sigma uncertantiies of magnitude of object 2

    Returns
    -------
        summErr : float
            one sigma uncertantiies on m1+m2
    """
    a = np.exp(1.84207 * m2) * s1**2
    b = np.exp(1.84207 * m1) * s2**2
    c = (np.exp(0.921084 * m1) + np.exp(0.921084 * m2)) ** 2
    return (a + b) / c


class population:
    def __init__(
        self,
        isoPaths: Union[str, Sequence[str]],
        alpha: float,
        bf: float,
        targetMass: float,
        ager: populationAgeDistribution,
        minMass: float = 0.2,
        maxMass: float = 2,
        bolometricCorrectionTables: Union[str, Sequence[str]] = "GAIA",
        distance: float = 0,
        colorExcess: float = 0,
        Rv: float = 3.1,
        artStar: Union[artificialStar, None] = None,
        pbar: bool = True,
    ):
        # TODO: Add default arguments so that most of these do not have do be set everytime
        self._logger = get_logger("fidanka.population.synthesize.population")

        self.distance: float = distance
        self.reddening: float = colorExcess
        self.Av: float = Rv * colorExcess
        self.Rv: float = Rv
        self.mu: float = 5 * np.log10(distance) - 5
        self.alpha: float = alpha
        self.bf: float = bf
        self.minMass: float = minMass
        self.maxMass: float = maxMass
        self.targetMass: float = targetMass

        self.artStar: Union[artificialStar, None] = artStar

        self._hasData: bool = False
        self._data: Union[npt.NDArray, None] = None
        self._totalMass: Union[float, None] = None

        self.pbar: bool = pbar

        self.age = ager

        self.iso, self.isoMeta = self._clean_input_isos(isoPaths)
        self.ages = np.array(list(self.iso[0].keys()))

        self._bolometricCorrectors = [
            BolometricCorrector(bolometricCorrectionTables, meta["[Fe/H]"])
            for meta in self.isoMeta
        ]

        # overwrite the theoretical isochrone with a bolometrically corrected version
        self.isoNP: Dict[int, Dict[float, npt.NDArray]] = dict()
        self._run_bolometric_corrections()

        self.header = list(self.iso[0][self.ages[0]].columns)

    @property
    def bcFilters(self):
        filters = list()
        for bc in self._bolometricCorrectors:
            filters.extend(bc.filters)
        return list(set(filters))

    def _run_bolometric_corrections(self):
        for isoID, (iso, bc) in tqdm(
            enumerate(zip(self.iso, self._bolometricCorrectors)),
            total=len(self._bolometricCorrectors),
            disable=not self.pbar,
            desc="Bolometrically Correcting All Populations",
        ):
            self.isoNP[isoID] = dict()
            for age, df in tqdm(
                iso.items(),
                disable=not self.pbar,
                desc=f"Correcting isochrones in pop {isoID}",
                leave=False,
            ):
                self._logger.info(
                    f"Calculating bolometric corrections for {age:0.2E} yr isochrone..."
                )
                mags = bc.apparent_mags(
                    10 ** df["log_Teff"],
                    df["log_g"],
                    df["log_L"],
                    Av=self.Av,
                    Rv=self.Rv,
                    mu=self.mu,
                )
                bolCorrectedIso = pd.concat([df, mags], axis=1)
                self.iso[isoID][age] = bolCorrectedIso
                self.isoNP[isoID][age] = bolCorrectedIso.values

    @staticmethod
    def _clean_input_isos(
        isoPaths: Union[str, Sequence[str]]
    ) -> Tuple[List[pd.DataFrame], List[Dict[str, float]]]:
        _logger = get_logger(
            "fidanka.population.synthesize.population._clean_input_isos"
        )
        iso, isoMeta = None, None
        if isinstance(isoPaths, str):
            _logger.info("Single isochrone provided to population synthesizer")
            iso = [read_iso(isoPaths)]
            isoMeta = [read_iso_metadata(isoPaths)]
        elif isinstance(isoPaths, Sequence):
            _logger.info("Multiple isochrones provided to population synthesizer")
            iso = [read_iso(isoFile) for isoFile in isoPaths]
            isoMeta = [read_iso_metadata(isoFile) for isoFile in isoPaths]

        return iso, isoMeta

    def _sample(self, age: float, popIndex: int, binary=False, mass=None, level=0):
        # TODO: Optimize age interpolation, this likeley can be cached to
        # save interpolaion time.
        younger, older = closest(self.ages, age)
        ageKeys = list(self.isoNP[popIndex].keys())
        if younger == None:
            younger, older = ageKeys[0], ageKeys[1]
        if older == None:
            younger, older = ageKeys[-2], ageKeys[-1]
        youngerIso = self.isoNP[popIndex][younger]
        olderIso = self.isoNP[popIndex][older]
        # TODO: fix typing here.
        isoAtAge = interpolate_keyed_arrays(
            youngerIso, olderIso, age, younger, older, key=0
        )  # EEPs in col 0

        massMap = isoAtAge[:, 2]
        mMin, mMax = massMap.min(), massMap.max()
        sortedMasses = np.sort(massMap)
        if mass is None:
            mass = sample_n_masses(1, self.alpha, mMin=mMin, mMax=mMax)[0]
        lowerMass, upperMass = closest(massMap, mass)
        if lowerMass == None:
            lowerMass = mMin
            upperMass = sortedMasses[1]
            self._logger.info("Falling back on end of array for lower mass")
            self._logger.info(f"Using masses {lowerMass} and {upperMass}")
        if upperMass == None:
            upperMass = mMax
            lowerMass = sortedMasses[-2]
            self._logger.info("Falling back on end of array for upper mass")
            self._logger.info(f"Using masses {lowerMass} and {upperMass}")
        isoBelowMass = isoAtAge[massMap == lowerMass]
        isoAboveMass = isoAtAge[massMap == upperMass]
        isoAtMass = interpolate_arrays(
            isoBelowMass, isoAboveMass, mass, lowerMass, upperMass
        )[0]

        # Sample Secondary if binary
        if binary:
            qMin = mMin / mass
            q = np.random.uniform(qMin, 1, 1)
            primaryMass = mass
            secondaryMass = q * primaryMass
            secondary, sm = self._sample(
                age, popIndex, binary=False, mass=secondaryMass[0], level=level + 1
            )
            assert sm == secondaryMass[0], "Mass Inconsistency in secondary!"
            secondaryParsed = np.array([x[1] for x in secondary.items()])
            isoAtMass += secondaryParsed
            for colID, (colAVal, colBVal) in enumerate(zip(isoAtMass, secondaryParsed)):
                isoAtMass[colID] = sum_mag(colAVal, colBVal)
            totalMass = mass + sm
        else:
            totalMass = mass

        outputPhotometry = dict()
        if self.artStar is not None and level == 0:
            for columnID, columnName in enumerate(self.header):
                if columnName in self.artStar:
                    truePhotometry = isoAtMass[columnID]
                    scale = self.artStar.err(truePhotometry, columnName)
                    perturbation = np.random.normal(scale=scale, loc=0, size=1)[0]
                    observedPhotometry = truePhotometry + perturbation
                    outputPhotometry[columnName] = observedPhotometry
        else:
            outputPhotometry = {
                key: value for key, value in zip(self.header, isoAtMass)
            }
        return outputPhotometry, totalMass

    def data(
        self,
        force=False,
        ageCacheSize: int = 1000,
        completnessMagName: Union[str, None] = None,
    ) -> Tuple[npt.NDArray, float]:
        if not self._hasData or force:
            ages = self.age.sample(ageCacheSize)

            totalMass = 0
            id = 0
            samples = list()
            with tqdm(desc="Sampling", disable=not self.pbar) as pbar:
                while totalMass <= self.targetMass:
                    isBinary = np.random.choice([True, False], p=[self.bf, 1 - self.bf])
                    whichPop = np.random.randint(0, len(self.isoNP), 1)[0]
                    photometry, mass = self._sample(
                        ages[id % ageCacheSize - 10], whichPop, binary=isBinary
                    )
                    id += 1
                    if id >= ageCacheSize - 10:
                        ages = self.age.sample(ageCacheSize)
                    totalMass += mass
                    samples.append(photometry)
                    pbar.update(1)
                    if id > 1000 and id % 1000 == 0:
                        pbar.total = int(
                            np.ceil(
                                id + ((self.targetMass - totalMass) / (totalMass / id))
                            )
                        )
                        pbar.refresh()
            if completnessMagName is not None and self.artStar is not None:
                survivingStars = list()
                for star in tqdm(samples):
                    p = np.random.uniform(0, 1)
                    completnessCheck = self.artStar.completness(
                        star[completnessMagName], completnessMagName
                    )
                    if p < completnessCheck:
                        survivingStars.append(star)

                self._data = survivingStars
            else:
                self._data = samples
            self._hasData = True
            self._totalMass = totalMass
        else:
            samples = self._data
            totalMass = self._totalMass

        return survivingStars, samples, totalMass


if __name__ == "__main__":
    artStarPath = "/Users/tboudreaux/Downloads/NGC2808A.XYVIQ.cal.zpt"
    artStar = artificialStar(artStarPath, sep=r"\s+")
    artStar.add_filter_alias(["Vvega", "Ivega"], ["Bessell_V", "Bessell_I"])

    ager = populationAgeDistribution(12e9, 12e9, lambda x: (x - x) + 12e9)

    isoPath = "/Users/tboudreaux/programming/fidankaTestData/isochrones.txt"
    pop = population(
        isoPath,
        -1,
        0.12,
        3e4,
        ager,
        colorExcess=0.4,
        distance=10e3,
        artStar=artStar,
    )
    observedPhotometry, photometry, totalClusterMass = pop.data(
        completnessMagName="Bessell_V"
    )
    print(totalClusterMass, len(observedPhotometry), len(photometry))

    import matplotlib.pyplot as plt

    for star in photometry:
        starColor = star["Bessell_V"] - star["Bessell_I"]
        starMag = star["Bessell_V"]
        plt.plot(starColor, starMag, "or", alpha=0.025, markersize=1)
    for star in observedPhotometry:
        starColor = star["Bessell_V"] - star["Bessell_I"]
        starMag = star["Bessell_V"]
        plt.plot(starColor, starMag, "ok", markersize=1)
    plt.gca().invert_yaxis()

    plt.xlabel("B-I")
    plt.ylabel("B")
    plt.show()
