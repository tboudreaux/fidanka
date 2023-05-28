from fidanka.bolometric.load import load_bol_table, load_bol_table_metadata
from fidanka.bolometric.color import get_mags
from fidanka.misc.utils import closest, interpolate_arrays

import re
import os
import numpy as np
import pandas as pd

RKE = re.compile(r"Av=(\d+\.\d+):Rv=(\d+\.\d+)")

class BolometricCorrector:
    def __init__(self, paths, FeH):
        self.paths = paths
        self.tables = dict()
        self.tableFeHs = np.empty(len(paths))
        for idx, path in enumerate(paths):
            tabFeH = re.search("feh([mp]\d+)", path).group(1)
            if tabFeH[0] == 'm':
                sign = -1
            else:
                sign = 1
            tabFehNS = tabFeH[1:]
            decimal = sign * float(f"{tabFehNS[0]}.{tabFehNS[1:]}")
            self.tableFeHs[idx] = decimal
        sortedPaths = [x for _,x in sorted(zip(self.tableFeHs, paths))]
        self.tableFeHs = np.sort(self.tableFeHs)
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

        self.Av = np.empty(len(self.upperBCTable))
        self.Rv = np.empty(len(self.upperBCTable))
        self.keys = list()
        for idx, reddeningKey in enumerate(self.upperBCTable):
            self.Av[idx] = reddeningKey[0]
            self.Rv[idx] = reddeningKey[1]
            self.keys.append(reddeningKey)


        # BC = interpolate_arrays(lowerBCTable.values(), upperBCTable.values(), FeH, closestFeHBelow, closestFeHAbove)
        # self.BC = pd.DataFrame(data=BC, columns=upperBCTable.columns)

        self.FeH = FeH

    def get_Av_correct_tables(self, table, Av, Rv=3.1):
        lowerAv, upperAv = closest(self.Av, Av)
        if lowerAv == None:
            lowerAv = self.Av[0]
            upperAv = self.Av[1]
        elif upperAv == None:
            lowerAv = self.Av[-2]
            upperAv = self.Av[-1]
        if lowerAv == upperAv:
            reddeningKey = (Av,Rv)
            return table[reddeningKey]

        lowerKey = (lowerAv, Rv)
        upperKey = (upperAv, Rv)
        lowerTable = table[lowerKey]
        upperTable = table[upperKey]
        interpolatedTable = interpolate_arrays(lowerTable.values, upperTable.values, Av, lowerAv, upperAv)
        interpolatedTable = pd.DataFrame(data=interpolatedTable, columns=upperTable.columns)
        return interpolatedTable

    def apparent_mags(self, Teff, logg, logL, Av=0, Rv=3.1, mu=0):
        # Get the Tables with the correct Av from the upper and lower bounding
        # metallicity tables
        AvCorrectUpperTable = self.get_Av_correct_tables(self.upperBCTable, Av, Rv)
        AvCorrectLowerTable = self.get_Av_correct_tables(self.lowerBCTable, Av, Rv)
        header = AvCorrectUpperTable.columns


        # Interpolate between the upper and lower bounding metallicity tables
        # to get the BC for the target metallicity
        targetBC = interpolate_arrays(
                AvCorrectLowerTable.values,
                AvCorrectUpperTable.values,
                self.FeH,
                self.FeHBounds[0],
                self.FeHBounds[1]
                )
        targetBC = pd.DataFrame(data=targetBC, columns=header)

        # get the magnitudes corrected for interstellar reddening
        dustCorrectedMags = get_mags(Teff, logg, logL, targetBC)

        # get the magnitudes corrected for distance modulus
        dustDistCorectedMags = {
                filterName : mag + mu
                for filterName, mag in dustCorrectedMags.items()
                }

        return pd.DataFrame(dustDistCorectedMags)

    def __repr__(self):
        return f"BolometricCorrector([Fe/H] : {self.FeH})"


if __name__ == "__main__":
    root = "/home/tboudreaux/d/Astronomy/GraduateSchool/Thesis/GCConsistency/NGC2808/bolTables/HSTWFC3/"
    filenames = list(filter(lambda x: re.search("feh[mp]\d+", x), os.listdir(root)))
    paths = list(map(lambda x: os.path.join(root, x), filenames))
    bol = Bolometric(paths, 1.0)

    print(bol.apparent_mags(5000, 2.0, 3.0, Av=0.156, mu=1))
