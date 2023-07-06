from fidanka.isochrone.isochrone import shift_full_iso
from fidanka.population.utils import sample_n_masses
from fidanka.bolometric import BolometricCorrector
from fidanka.misc.utils import closest, interpolate_arrays, get_samples, get_logger
from fidanka.isochrone.MIST import read_iso, read_iso_metadata


import pickle as pkl
import numpy as np
import pandas as pd
import re

import numpy.typing as npt

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


# TODO: Change this to use cluster mass instead of total cluster membership
# Note, that this is not observed cluster mass, rather it is total cluster
# mass. this means that completness should come at the very end instead of
# as the begining.


# TODO: Find a more elegant manner to deal with age. Should there be an age
# object? One which may include both the PDF as well as the min and max
# age? It seems very inellegant to have to pass those three independently.
class population:
    def __init__(
        self,
        isoPaths: Union[str, Sequence[str]],
        alpha: float,
        bf: float,
        agePDF,
        n,
        minAge,
        maxAge,
        minMass: float,
        maxMass: float,
        artStarFuncs,
        distance: float,
        colorExcess: float,
        magName,
        bolometricCorrectionTables: Union[str, Sequence[str]],
        Rv=3.1,
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

        self.age = agePDF
        self.n = n
        self.minAge = minAge
        self.maxAge = maxAge

        self.iso, self.isoMeta = self._clean_input_isos(isoPaths)
        self.ages = np.array(list(self.iso[0].keys()))

        self._bolometricCorrectors = [
            BolometricCorrector(bolometricCorrectionTables, meta["[Fe/H]"])
            for meta in self.isoMeta
        ]

        # overwrite the theoretical isochrone with a bolometrically corrected version
        self.isoNP = list()
        for isoID, (iso, bc) in enumerate(zip(self.iso, self._bolometricCorrectors)):
            for age, df in iso.items():
                self._logger.info(
                    f"Calculating bolometric corrections for {age} Gyr isochrone..."
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
                self.isoNP.append(bolCorrectedIso.values)

        self.header = list(self.iso[0][self.ages[0]].columns)

        # TODO: This is where the code is going to need to have major changes in order
        # to comply with the new artificial star obect
        self._goodFilterIdx = list()
        self.noiseFuncs = dict()
        self._effectiveWavelengths = list()
        self._nonFilterIndices = list()
        self._goodColumnNames = list()
        self._completness = artStarFuncs.pop("Completness")
        for cID, column in enumerate(self.header):
            match = re.search(FILTERPATTERN, column)
            if match:
                filterName = match.group(1)
                if filterName in artStarFuncs:
                    self.noiseFuncs[cID] = artStarFuncs[filterName]
                    self._goodFilterIdx.append(cID)
                    self._effectiveWavelengths.append(float(filterName[1:-1]))
                    self._goodColumnNames.append(f"{filterName}")
            else:
                self._nonFilterIndices.append(cID)

        self._hasData = False
        self._data = None
        self._completnessCheckColumnID = self._goodColumnNames.index(magName)
        self._completnessCheckColName = self._goodColumnNames[
            self._completnessCheckColumnID
        ]
        self._isoFilterCompletenessCheckColName = (
            f"ACS_WFC_{self._completnessCheckColName}_MAG"
        )
        self._mappingFunctions = self._generate_mapping_functions()

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

        # if isinstance(iso, None) or isinstance(isoMeta, None):
        #     _logger.error("Unable to load isochrones!")
        #     raise RuntimeError(f"Unable to load {isoPaths} properly")

        return iso, isoMeta

    # TODO: This function is likely not needed anymore. Assuming all completness
    # estimates can be moved to the end of the population synthethis.
    def _generate_mapping_functions(self):
        mappingFunctions = list()
        for isoID, iso in enumerate(self.iso):
            self._logger.info(f"Generating mapping functions for isochrone {isoID}")
            validAges = np.array(list(iso.keys()))
            sortedMasses = list()
            maxSize = 0
            for age in validAges:
                sortedMasses.append(np.sort(iso[age]["initial_mass"].values))
                maxSize = max(maxSize, len(sortedMasses[-1]))

            sortedAges = np.sort(validAges)

            apparentMags = np.empty((len(sortedAges), maxSize))
            for ageID, age in enumerate(sortedAges):
                print(
                    f"Age Point: {ageID}, with {len(sortedMasses[ageID])} to work on",
                    end="",
                )
                for massID, mass in enumerate(sortedMasses[ageID]):
                    isoAtAge = iso[age]
                    isoAtAgeAndMass = isoAtAge[isoAtAge["initial_mass"] == mass]
                    extracted = isoAtAgeAndMass[
                        self._isoFilterCompletenessCheckColName
                    ].values
                    if extracted.size == 0:
                        apparentMags[ageID, massID] = np.nan
                    else:
                        apparentMags[ageID, massID] = extracted[0]
                print("\r", end="")
            print("")

            num_pairs = sum([len(mass_list) for mass_list in sortedMasses])

            input_coords = np.empty((num_pairs, 2))
            output_values = np.empty(num_pairs)

            index = 0
            for i, age in enumerate(sortedAges):
                for j, mass in enumerate(sortedMasses[i]):
                    input_coords[index] = [age, mass]
                    output_values[index] = apparentMags[i][j]
                    index += 1

            Z = LinearNDInterpolator(input_coords, output_values)
            mappingFunctions.append(Z)
        return mappingFunctions

    def _sample_new(self, age, popIndex, binary=False, mass=None, level=0):
        # TODO: Optimize age interpolation, this likeley can be cached to
        # save interpolaion time.
        younger, older = closest(self.ages, age)
        youngerIso = self.isoNP[popIndex][younger]
        olderIso = self.isoNP[popIndex][older]
        # TODO: fix typing here.
        isoAtAge = interpolate_eep_arrays(youngerIso, olderIso, age, younger, older)

        massMap = isoAtAge[:, 2]
        mMin, mMax = massMap.min(), massMap.max()
        sortedMasses = np.sort(massMap)
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
        isoBelowMass = self.isoNP[massMap == lowerMass]
        isoAboveMass = self.isoNP[massMap == upperMass]
        isoAtMass = interpolate_arrays(
            isoBelowMass, isoAboveMass, mass, lowerMass, upperMass
        )[0]

        # Sample Secondary if binary
        if binary:
            qMin = mMin / mass
            q = np.random.uniform(qMin, 1, 1)
            primaryMass = mass
            secondaryMass = q * primaryMass
            secondary = self._sample_new(
                age, popIndex, binary=False, mass=secondaryMass, level=level + 1
            )

    def _sample(self, age, idx, popI, samples, binary=False, mass=None):
        isPrimary = False
        isSingle = False
        if binary and mass is None:
            isPrimary = True
        if not binary and mass is None:
            isSingle = True
        younger, older = closest(self.ages, age)
        youngerIso = self.isoNP[popI][younger]
        olderIso = self.isoNP[popI][older]
        isoAtAge = interpolate_eep_arrays(youngerIso, olderIso, age, younger, older)

        # TODO: Use the BolometricCorrector to get the bolometric correction

        massMap = isoAtAge[:, 2]
        sortedMasses = np.sort(massMap)

        # TODO: Use bolometric corrector here
        magMap = isoShiftedToDistRed[:, self._completnessCheckColumnID]

        completnessMap = interp1d(
            massMap, magMap, kind="linear", bounds_error=False, fill_value=np.nan
        )

        # TODO: Use artificial star object here
        completness = lambda m, alpha: self._completness(completnessMap(m))

        resetMass = False
        mMin = massMap.min()
        mMax = massMap.max()
        if mass is None:
            resetMass = True
            mass = sample_n_masses(1, completness, self.alpha, mMin, mMax)[0]
        else:
            if isinstance(mass, Sequence):
                mass = mass[0]
        lowerMass, upperMass = closest(isoAtAge[:, 2], mass)
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

        # TODO: Updated to use the bolometric corrector directly
        lowerMassPoint = isoShiftedToDistRed[isoAtAge[:, 2] == lowerMass]
        upperMassPoint = isoShiftedToDistRed[isoAtAge[:, 2] == upperMass]

        targetSample = interpolate_arrays(
            lowerMassPoint, upperMassPoint, mass, lowerMass, upperMass
        )[0]

        for rID, (ID, interpFunc) in enumerate(self.noiseFuncs.items()):
            samples[idx][2 + rID] = sum_mag(targetSample[rID], samples[idx][2 + rID])
            if isSingle:
                scale = interpFunc(targetSample[rID])
                dist = np.random.normal(loc=0, scale=scale, size=1)
                samples[idx][2 + rID] += dist
                samples[idx][2 + len(self.noiseFuncs) + rID] = scale

        if isPrimary:
            # Based on Milone et al. 2012 (A&A 537, A77)
            # Assume a flat mass ratio distribution
            qMin = mMin / mass
            q = np.random.uniform(qMin, 1, 1)
            primaryMass = mass
            secondaryMass = q * primaryMass
            self._sample(age, idx, popI, samples, binary=False, mass=secondaryMass)
            # Inject noise into binary system after summing the magnitudes
            for rID, (ID, interpFunc) in enumerate(self.noiseFuncs.items()):
                scale = interpFunc(targetSample[rID])
                dist = np.random.normal(loc=0, scale=scale, size=1)
                samples[idx][2 + rID] += dist
                samples[idx][2 + len(self.noiseFuncs) + rID] = scale
            samples[idx][0] = primaryMass
            samples[idx][-2] = q
            samples[idx][1] = (age + samples[idx][1]) / 2
        else:
            if resetMass:
                samples[idx][0] = mass
            samples[idx][1] = age
            samples[idx][-2] = 1

        samples[idx][-1] = binary

    def data(self, force=False):
        if not self._hasData or force:
            ages = get_samples(
                self.n, self.age, domain=np.linspace(self.minAge, self.maxAge, 1000)
            )
            samples = np.zeros((self.n, 2 * len(self._goodFilterIdx) + 4))
            for filterID, _ in enumerate(self._goodFilterIdx):
                samples[:, 2 + filterID] = np.inf

            whichPop = np.random.randint(0, len(self.isoNP), self.n)

            for idx, (age, popI) in enumerate(zip(ages, whichPop)):
                isBinary = np.random.choice([True, False], p=[self.bf, 1 - self.bf])
                self._sample(age, idx, popI, samples, binary=isBinary)
            self._data = samples
            self._hasData = True
        else:
            samples = self._data

        return samples

    def _resample_binaries(self):
        if self._hasData and self._data is not None:
            nb = self.bf * self.n
            ns = (1 - self.bf) * self.n
            cns = np.sum(self._data[:, -1] == 0)
            dns = ns - cns
            rIndexies = np.random.choice(
                np.argwhere(self._data[:, -1] == 0).flatten(), size=dns, replace=False
            )
            if cns > 0:
                ages = get_samples(
                    dns, self.age, domain=np.linspace(self.minAge, self.maxAge, 1000)
                )
                samples = np.zeros((dns, 2 * len(self._goodFilterIdx) + 4))
                whichPop = np.random.randint(0, len(self.isoNP), dns)

                for idx, (rIDX, age, popI) in enumerate(zip(rIndexies, ages, whichPop)):
                    self._sample(age, idx, popI, samples, binary=False)

                for idx, rIDX in enumerate(rIndexies):
                    self._data[rIDX] = samples[idx]
            elif cns < 0:
                self._data = np.delete(self._data, rIndexies, axis=0)
            else:
                pass
            bIdx = np.argwhere(self._data[:, -1] == 1).flatten()
            self._data = np.delete(self._data, bIdx, axis=0)

            ages = get_samples(
                nb, self.age, domain=np.linspace(self.minAge, self.maxAge, 1000)
            )
            samples = np.zeros((nb, 2 * len(self._goodFilterIdx) + 4))
            whichPop = np.random.randint(0, len(self.isoNP), nb)

            for idx, (age, popI) in enumerate(zip(ages, whichPop)):
                self._sample(age, idx, popI, samples, binary=True)

            self._data = np.concatenate((self._data, samples), axis=0)

    def resample(self, bf=None):
        if not self._hasData:
            raise ValueError("No data to resample")
        if bf is not None:
            self.bf = bf
            self._resample_binaries()
        assert self._data is not None
        return self._data

    def to_pandas(self):
        columnNames = (
            ["Mass", "Age"]
            + self._goodColumnNames
            + [f"{name}_err" for name in self._goodColumnNames]
            + ["q", "isBinary"]
        )
        df = pd.DataFrame(self.data(), columns=columnNames)
        return df

    def to_csv(self, filename):
        df = self.to_pandas()
        df.to_csv(filename, index=False)
