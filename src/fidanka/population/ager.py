from fidanka.misc.utils import closest, interpolate_arrays, get_samples, get_logger

import numpy as np


class populationAgeDistribution:
    def __init__(self, minAge, maxAge, agePDF):
        self._minAge = minAge
        self._maxAge = maxAge
        self._agePDF = agePDF
        self._logger = get_logger("fidanka.population.ager.populationAgeDistribution")
        self._logger.info(
            f"Initialized ager with minAge: {minAge} and maxAge: {maxAge}"
        )

    @property
    def min(self):
        return self._minAge

    @property
    def max(self):
        return self._maxAge

    def sample(self, n, domainSize=100):
        samples = get_samples(
            n,
            self._agePDF,
            domain=np.linspace(self.min, self.max, domainSize),
        )
        return samples
