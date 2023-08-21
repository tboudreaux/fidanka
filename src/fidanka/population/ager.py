from collections.abc import Callable
from fidanka.misc.utils import closest, interpolate_arrays, get_samples, get_logger

import numpy as np
from numpy import typing as npt
from typing import Callable

FARRAY_1D = npt.NDArray


class populationAgeDistribution:
    def __init__(
        self, minAge: float, maxAge: float, agePDF: Callable[[FARRAY_1D], FARRAY_1D]
    ):
        """
        age object to make population syththesis a little more clean. This object
        takes care of all things related to the sampled age distribution of stars
        in the GC.

        Parameters
        ----------
            minAge : float
                The minium age which will ever be sampled
            maxAge : float
                The maxium age which will ever be sampled
            agePDF : Callable[[np.ndarray], np.ndarray]
                Normalized probability age distribution which will be sampled from.
        """
        self._minAge: float = minAge
        self._maxAge: float = maxAge
        self._agePDF: Callable[[FARRAY_1D], FARRAY_1D] = agePDF

        self._logger = get_logger("fidanka.population.ager.populationAgeDistribution")
        self._logger.info(
            f"Initialized ager with minAge: {minAge} and maxAge: {maxAge}"
        )

    @property
    def min(self) -> float:
        """
        Get the minium age defined at ager instantiation time

        Returns
        -------
            minAge : float
                The minium age which will be sampled
        """
        return self._minAge

    @property
    def max(self) -> float:
        """
        Get the maximum age defined at ager instantiation time

        Returns
        -------
            maxAge : float
                The maximum age which will be sampled
        """
        return self._maxAge

    def sample(self, n: int, domainSize: int = 100) -> FARRAY_1D:
        """
        Sample the age probibility distribution provided at ager instantiation time

        Parameters
        ----------
            n : int
                number of samples to generate
            domainSizez : int, default=100
                size of the sampling domain. Larger numbers will lead to a smoother sampling
                of the age PDF

        Returns
        -------
            samples : FARRAY_1D
                array of sampled ages which will be of size n
        """
        samples = get_samples(
            n,
            self._agePDF,
            domain=np.linspace(self.min, self.max, domainSize),
        )

        return samples
