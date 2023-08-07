from collections.abc import Sequence
from warnings import filters
from fidanka.misc.utils import get_logger

from typing import Callable, Union, List, Dict, Tuple

from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
import difflib


class artificialStar:
    def __init__(self, calibrated_file: Union[str, None] = None, **kwargs):
        self._logger = get_logger("fidanka.population.artificialStar")
        self.generated = False
        self._df = None
        self._filters: List[str] = list()
        self._err_funcs: Dict[str, Callable[[float], float]] = dict()
        self._completness_functions: Dict[str, Callable[[float], float]] = dict()
        self._alias: Dict[str, List[str]] = dict()

        if calibrated_file:
            self.from_calibrated_file(calibrated_file, **kwargs)
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
                binMeanMag,
                binMeanCompletness,
                bounds_error=False,
                fill_value="extrapolate",
            )

    def _resolve_filter_name(self, name: str, strict: bool = False) -> str:
        """
        Given a filter name (which may either be in the artificial star
        test file or an alias defined by the user) resolve this to a name
        which is actually in the artificial star test.

        Parameters
        ----------
            name : str
                name of filter, either the filter name from the artificial star
                test file or a user defined alias
            strict : bool, default=False
                If true, raise a KeyError if the filter name is unable to be
                resolved using a fuzzy search.

        Returns
        -------
            trueName : str
                Resolved name as it appears in the artificial star test file
                column names.
            False
                if name is unable to be resolved
        """
        for trueName, aliasNames in self.aliases.items():
            if name in aliasNames:
                return trueName
            elif name not in aliasNames and not strict:
                ...
        raise KeyError(f"Unable to resolve filter name {filter}")

    def __contains__(self, key):
        try:
            self._resolve_filter_name(key)
        except KeyError:
            return False
        return True

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
        resolvedFilterName = self._resolve_filter_name(filter)
        return self._err_funcs[resolvedFilterName](mag)

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
        resolvedFilterName = self._resolve_filter_name(filter)
        return self._completness_functions[resolvedFilterName](mag)

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

    @property
    def aliases(self) -> Dict[str, List[str]]:
        return self._alias

    def add_filter_alias(
        self,
        filterNames: Union[str, Sequence[str]],
        aliasNames: Union[str, Sequence[str]],
    ) -> Dict[str, List[str]]:
        """
        Add an alias to a filter name which can be used by other functions when looking
        to match photometric filters

        Parameters
        ----------
            filterName : Union[str, Sequence[str]]
                List of, or single, filter name(s) (as auto identified, which may be checked with self.filters)
                which you which to add an alias to.
            aliasNames : Union[str, Sequenace[str]]
                List of alias which you wish to add. Can only add one alias to a filter at a time but may add aliass to multiple filters simultaniously. That is to sat if you have set filterNames = ["Vvege", "Ivega"] then aliasNames may be set to ["F606W", "F814W"].

        Returns
        -------
            aliases : Dict[str, List[str]]
                A dictionary defining all the currently defined aliases. This can also be checked
                using self.aliases
        """
        filterNames = [filterNames] if isinstance(filterNames, str) else filterNames
        aliasNames = [aliasNames] if isinstance(aliasNames, str) else aliasNames

        for filterName, alias in zip(filterNames, aliasNames):
            self._alias[filterName].append(alias)

        return self.aliases

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
                self._alias[columnName] = [columnName]
                self._err_funcs[columnName] = interp1d(
                    self._df[columnName].values,
                    self._df[followingColumn].values,
                    bounds_error=False,
                    fill_value="extrapolate",
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
    artStar.add_filter_alias("Vvega", "F606W")
    print(artStar.aliases)
