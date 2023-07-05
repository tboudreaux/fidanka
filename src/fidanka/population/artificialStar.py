from fidanka.misc.utils import get_logger

from typing import Callable, Union, List, Dict

from scipy.interpolate import interp1d
import pandas as pd


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

    def completness(self, mag: float, filter: str) -> float:
        ...

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
