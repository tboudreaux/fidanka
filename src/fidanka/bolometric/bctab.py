from fidanka.bolometric.load import load_bol_table, load_bol_table_metadata
from fidanka.bolometric.color import get_mags
from fidanka.misc.utils import closest, interpolate_arrays
from fidanka.misc.utils import get_logger

import re
import os
import numpy as np
import pandas as pd

from hashlib import sha256

RKE = re.compile(r"Av=(\d+\.\d+):Rv=(\d+\.\d+)")


class BolometricCorrector:
    def __init__(self, paths, FeH, filters=["F606W", "F814W"]):
        self.logger = get_logger("fidanka.bolometric.BolometricCorrector")

        self.filters = filters
        self.paths = paths
        self.tables = dict()
        self.tableFeHs = np.empty(len(paths))
        self.FeH = FeH

        self.logger.info("Loading FeH Tables...")
        for idx, path in enumerate(paths):
            tabFeH = re.search(r"feh([mp]\d+)", path).group(1)
            if tabFeH[0] == 'm':
                sign = -1
            else:
                sign = 1
            tabFehNS = tabFeH[1:]
            decimal = sign * float(f"{tabFehNS[0]}.{tabFehNS[1:]}")
            self.tableFeHs[idx] = decimal
        self.logger.info("FeH tables loaded!")

        sortedPaths = [x for _,x in sorted(zip(self.tableFeHs, paths))]
        self.tableFeHs = np.sort(self.tableFeHs)

        self.logger.info("Interperloating FeH Tables...")
        closestFeHBelow, closestFeHAbove = closest(self.tableFeHs, FeH)
        if closestFeHAbove == None:
            closestFeHAbove = self.tableFeHs[-1]
            closestFeHBelow = self.tableFeHs[-2]
        elif closestFeHBelow == None:
            closestFeHBelow = self.tableFeHs[0]
            closestFeHAbove = self.tableFeHs[1]
        self.FeHBounds = (closestFeHBelow, closestFeHAbove)
        lowerBCTablePath = sortedPaths[np.where(self.tableFeHs == closestFeHBelow)[0][0]]
        upperBCTablePath = sortedPaths[np.where(self.tableFeHs == closestFeHAbove)[0][0]]

        self.upperBCTable = load_bol_table(upperBCTablePath)
        self.lowerBCTable = load_bol_table(lowerBCTablePath)
        self.header = self.upperBCTable[list(self.upperBCTable.keys())[0]].columns
        self.fullFilterNames = [
                x for x in self.header if any([fn in x for fn in self.filters])
                ]
        self.filterKeyIDs = [
                self.header.get_loc(i)
                for i in self.fullFilterNames
                if i in self.header
                ]
        self.filterKeyIDs.insert(0,1)
        self.filterKeyIDs.insert(0,0)

        self.BCTabs = dict()
        for (lKey, lower), (uKey, upper) in zip(self.lowerBCTable.items(), self.upperBCTable.items()):
            self.BCTabs[lKey] = interpolate_arrays(
                    lower,
                    upper,
                    self.FeH,
                    self.FeHBounds[0],
                    self.FeHBounds[1]
                    )
        self.logger.info("FeH Tables Interperolated!")

        self.logger.info("Intializing Cache...")
        self.Av = np.empty(len(self.upperBCTable))
        self.Rv = np.empty(len(self.upperBCTable))
        self.keys = list()
        for idx, reddeningKey in enumerate(self.upperBCTable):
            self.Av[idx] = reddeningKey[0]
            self.Rv[idx] = reddeningKey[1]
            self.keys.append(reddeningKey)

        self._cache ={
                'AvCorrectUpperTable' : None,
                'AvCorrectLowerTable' : None,
                'targetBC' : None,
                'dustCorrectedMags' : None,
                'dustDistCorectedMags' : None
                }
        self._cacheHash = None
        self._cacheHits = 0
        self._cacheMisses = 0
        self.logger.info("Cache Initialized!")

    def _check_cache(self, Av, Rv):
        cacheHash = sha256(
                np.array([Av, Rv]).tobytes()
                ).hexdigest()
        if cacheHash == self._cacheHash:
            self._cacheHits += 1
            return True
        else:
            self._cacheHash = cacheHash
            self._cacheMisses += 1
            return False

    def _reset_cache(self):
        self._cache = {
                'targetBC' : None,
                }
        self._cacheHash = None

    def _update_cache_hash(self, Av, Rv):
        self._cacheHash = sha256(
                np.array([Av, Rv]).tobytes()
                ).hexdigest()


    def apparent_mags(self, Teff, logg, logL, Av=0, Rv=3.1, mu=0):
        # Get the Tables with the correct Av from the upper and lower bounding
        # metallicity tables
        if self._check_cache(Av, Rv):
            targetBC = self._cache['targetBC']
        else:
            self._reset_cache()
            self._update_cache_hash(Av, Rv)

            lowerAv, upperAv = closest(self.Av, Av)
            upperKey, lowerKey = (upperAv, Rv), (lowerAv, Rv)
            upperAvTab = self.BCTabs[upperKey]
            lowerAvTab = self.BCTabs[lowerKey]

            if upperKey == lowerKey:
                targetBC = upperAvTab
            else:
                targetBC = interpolate_arrays(
                        lowerAvTab,
                        upperAvTab,
                        Av,
                        lowerAv,
                        upperAv
                        )

            targetBC = targetBC[:, self.filterKeyIDs]
            self._cache['targetBC'] = targetBC

        # get the magnitudes corrected for interstellar reddening
        try:
            dustCorrectedMags = get_mags(Teff, logg, logL, targetBC)
        except Exception as e:
            self.logger.error(f"Av: {Av}, Rv: {Rv}")
            self.logger.error(f"mu: {mu}")
            self.logger.error(f"self.FeH: {self.FeH}")
            self.logger.error(f"self.FeHBounds: {self.FeHBounds}")
            raise e

        # get the magnitudes corrected for distance modulus
        dustDistCorectedMags = {
                filterName : mag + mu
                for filterName, mag in zip(self.filters, dustCorrectedMags.T)
                }
        return dustDistCorectedMags

    def __repr__(self):
        return f"BolometricCorrector([Fe/H] : {self.FeH})"


if __name__ == "__main__":
    root = "/home/tboudreaux/d/Astronomy/GraduateSchool/Thesis/GCConsistency/NGC2808/bolTables/HSTWFC3/"
    filenames = list(filter(lambda x: re.search("feh[mp]\d+", x), os.listdir(root)))
    paths = list(map(lambda x: os.path.join(root, x), filenames))
    bol = BolometricCorrector(paths, 1.0)

    Teff = np.array([5000])
    logg = np.array([2.0])
    logL = np.array([3.0])

    print(bol.apparent_mags(Teff, logg, logL, Av=0.156, mu=1))
