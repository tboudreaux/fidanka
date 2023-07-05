from fidanka.misc.utils import get_logger

from typing import Callable, Union, List, Dict

from scipy.interpolate import interp1d
import pandas as pd
import difflib


class artificialStar:
    def __init__(self, calibrated_file: Union[str, None] = None):
        self._logger = get_logger("fidanka.population.artificialStar")
        self.generated = False
        self._df = None
        self._filters: List[str] = list()
        self._err_funcs: Dict[str, Callable] = dict()

        if calibrated_file:
            self.from_calibrated_file(calibrated_file)
        else:
            self._logger.info(
                "No calibrated artificial star file provided at instantiation time."
            )

        ...

    @property
    def shared_system_name(self) -> str:
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
        return sharedSystemName

    def _gen_completness_functions(self, threshold: float = 0.02):
        assert isinstance(
            self._df, pd.DataFrame
        ), "Unknown error, self._df not loaded properly"
        sharedSystemName = self.shared_system_name
        groundFilters = [x for x in self._df.columns if "ground" in x]
        matchedFilters = list()
        for theoreticalFilter in self.filters:
            for observedFilter in groundFilters:
                if theoreticalFilter.strip(sharedSystemName) == observedFilter.strip(
                    "ground"
                ):
                    matchedFilters.append((theoreticalFilter, observedFilter))
        for matchedFilter in matchedFilters:
            compName = f"{matchedFilter[0]} - {matchedFilter[1]}"
            self._df[compName] = (
                self._df[matchedFilter[0]] - self._df[matchedFilter[1]]
            ) / self._df[matchedFilter[0]]
            self._df[f"{matchedFilter[0]}recovered"] = (
                True if self._df[compName] < threshold else False
            )
        print(self._df)

    def err(self, mag: float, filter: str) -> float:
        assert (
            self.generated
        ), "Must pass calibrated artificial star test before evaluating errors."
        assert (
            filter in self._filters
        ), f"Requested filter {filter} not in list of identified filters {self._filters}"
        return self._err_funcs[filter](mag)

    @property
    def filters(self) -> List[str]:
        return self._filters

    def from_calibrated_file(self, path: str) -> bool:
        self._logger.info(f"Generating calibration functions from file: {path}")
        self._filters = list()
        self._df = pd.read_csv(path, sep=r"\s+")
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
    artStar.from_calibrated_file("~/Downloads/NGC2808A.XYVIQ.cal.zpt")
    print(artStar.filters)
    print(artStar.err(20, "Vvega"))
    print(artStar)
