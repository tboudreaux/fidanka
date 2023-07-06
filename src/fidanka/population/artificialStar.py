from warnings import filters
from fidanka.misc.utils import get_logger

from typing import Callable, Union, List, Dict

from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import difflib


class artificialStar:
    def __init__(self, calibrated_file: Union[str, None] = None):
        self._logger = get_logger("fidanka.population.artificialStar")
        self.generated = False
        self._df = None
        self._filters: List[str] = list()
        self._err_funcs: Dict[str, Callable[[float], float]] = dict()
        self._completness_functions: Dict[str, Callable[[float], float]] = dict()

        if calibrated_file:
            self.from_calibrated_file(calibrated_file)
        else:
            self._logger.info(
                "No calibrated artificial star file provided at instantiation time."
            )

    @property
    def shared_system_name(self) -> str:
        """
        Find the photometric system name assuming that they are encoded into the
        calibrated artificial star test column names like <A><system> (such as Ivega)
        The filter id but be one charecter and the system must follow it. The shared
        system name will then be the intersection between any identified filters

        Returns
        -------
            sharedSystemName : str
                The auto identified shared photometric system name.
        """
        differ = difflib.Differ()
        sharedSystemList = list()
        for filterA in self.filters:
            for filterB in self.filters:
                if filterA != filterB:
                    intersection = differ.compare(filterA, filterB)
                    sharedSystemList.append(list(intersection))

        sharedSystemNameSet = set(sharedSystemList[0])
        for intersection in sharedSystemList[1:]:
            sharedSystemNameSet = sharedSystemNameSet & set(intersection)

        ids = list()
        chars = list()
        for sharedSystemNameChar in sharedSystemNameSet:
            char = sharedSystemNameChar.lstrip()
            ids.append(self.filters[0].index(char))
            chars.append(char)
        sharedSystemName = "".join(
            map(lambda x: x[0], sorted(zip(chars, ids), key=lambda x: x[1]))
        )
        self._logger.info(f"Identified photometric system : {sharedSystemName}")
        return sharedSystemName

    def _gen_completness_functions(
        self,
        threshold: float = 0.02,
        xKey: str = "xsig",
        yKey: str = "ysig",
        binWidth: float = 0.05,
    ):
        assert isinstance(
            self._df, pd.DataFrame
        ), "Unknown error, self._df not loaded properly"

        r = np.sqrt(self._df[xKey].values ** 2 + self._df[yKey].values ** 2)
        self._df["completness"] = [1 if ri < threshold else 0 for ri in r]
        for filter in self.filters:
            filterPhotometry = self._df[filter]
            filterBinLeftEdges = np.arange(
                filterPhotometry.min(), filterPhotometry.max(), binWidth
            )
            filterBinRightEdges = filterBinLeftEdges + binWidth
            binMeanMag, binMeanCompletness = list(), list()
            for leftEdge, rightEdge in zip(filterBinLeftEdges, filterBinRightEdges):
                selectedAS = self._df[
                    (filterPhotometry >= leftEdge) & (filterPhotometry < rightEdge)
                ]
                if selectedAS.shape[0] > 0:
                    binMeanMag.append(leftEdge + (binWidth / 2))
                    binMeanCompletness.append(selectedAS["completness"].mean())

            binMeanMag, binMeanCompletness = np.array(binMeanMag), np.array(
                binMeanCompletness
            )

            sortedBinMag = np.argsort(binMeanMag)
            binMeanMag = binMeanMag[sortedBinMag]
            binMeanCompletness = binMeanCompletness[sortedBinMag]

            self._completness_functions[filter] = interp1d(
                binMeanMag, binMeanCompletness
            )

    def err(self, mag: float, filter: str) -> float:
        """
        Get the estimate photometric uncertainty from calibrated arificial star
        tests.

        Parameters
        ----------
            mag : float
                Apparent Magnitude
            filter : float
                Filter to work in. Avalible filters may be checked with the filters
                property of an instantiated artificial star class.

        Returns
        -------
            err : float
                estimated photometric uncertainty for <mag> in <filter>
        """
        assert (
            self.generated
        ), "Must pass calibrated artificial star test before evaluating errors."
        assert (
            filter in self._filters
        ), f"Requested filter {filter} not in list of identified filters {self._filters}"
        return self._err_funcs[filter](mag)

    def completness(self, mag: float, filter: str) -> float:
        """
        Get the estimated completness at a given magnitude in a given filte

        Parameters
        ----------
            mag : float
                Apparent Magnitude
            filter : float
                Filter to work in. Avalible filters may be checked with the filters
                property of an instantiated artificial star class.

        Returns
        -------
            completness : float
                Estimated fractional completness, ranging [0,1], for <mag> in <filter>
        """
        assert (
            self.generated
        ), "Must pass calibrated artificial star test before evaluating errors."
        assert (
            filter in self._filters
        ), f"Requested filter {filter} not in list of identified filters {self._filters}"
        return self._completness_functions[filter](mag)

    @property
    def filters(self) -> List[str]:
        """
        List filters which have been infered from the given calibrated artificial star
        test file.

        Returns
        -------
            filters : List[str]
                List of filters which have been infered. Filters are infered as any
                column which immediatly proceeds a column containing the string 'err'.
                This is not an incredibly general algorithm so make sure to preprocess
                your datafiles to work in accordance with it.
        """
        return self._filters

    def from_calibrated_file(self, path: str, **kwargs) -> bool:
        """
        Instantiate artificial star test from a calibratied file. If a path is provived
        at instantiation time this is the function which is called. This will
        infer filter names as the names of any columns immediatly suceeded by a column
        whoes name containes the string 'err'. Parsing is done with pandas read_csv
        method, all read_csv keywords are exposed through this.

        Parameters
        ----------
            path : str
                File path to calibrated artificial star file
            **kwargs
                See pandas read_csv

        Returns
        -------
            generated : bool
                True if generated correctly
        """
        self._logger.info(f"Generating calibration functions from file: {path}")
        self._filters = list()
        self._df = pd.read_csv(path, **kwargs)
        for columnName, followingColumn in zip(self._df.columns, self._df.columns[1:]):
            if "err" in followingColumn:
                self._filters.append(columnName)
                self._err_funcs[columnName] = interp1d(
                    self._df[columnName].values, self._df[followingColumn].values
                )
        self._gen_completness_functions()
        self.generated = True
        return self.generated

    def __repr__(self):
        out = f"<artificialStar - generated : {self.generated} - filters : {', '.join(self._filters)}>"
        return out


if __name__ == "__main__":
    artStar = artificialStar()
    artStar.from_calibrated_file("~/Downloads/NGC2808A.XYVIQ.cal.zpt", sep=r"\s+")
