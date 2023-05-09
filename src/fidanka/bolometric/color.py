from fidanka.bolometric.load import load_bol_table

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

REJECTNAMES = ["Teff", "logg", "[Fe/H]", "Av", "Rv"]
SOLBOL = 4.75

def _get_mags(Teff, logg, logL, table):
    """
    Get the magnitudes of a star given its Teff, logg, logL, and bolometric
    correction table.

    Parameters
    ----------
    Teff : float
        Effective temperature of the star in Kelvin.
    logg : float
        log10 of the surface gravity of the star in cgs units.
    logL : float
        log10 of the luminosity of the star in cgs units.
    table : pandas.DataFrame
        Bolometric correction table for a given metallicity, Av, and Rv.

    Returns
    -------
    magnitudes : dict
        Dictionary of magnitudes for each filter in the bolometric correction
    """
    filterNames = list(filter(lambda x: x not in REJECTNAMES, table.columns))
    magNames = map(lambda x: f"{x}_MAG", filterNames)
    magnitudes = dict()
    for magName, filterName in zip(magNames, filterNames):
        interpFunc = LinearNDInterpolator(list(zip(table["Teff"], table["logg"])), table[filterName])
        magnitudes[magName] = SOLBOL-2.5*logL - interpFunc(Teff, logg)
    return magnitudes


def get_interpolated_FeHTable(BC1, BC2, Av, Rv, FeH):
    """
    Get the bolometric correction table for a given metallicity, Av, and Rv. If
    that mettallicity is not in the BC tables, linearlly interpolate between
    the two closest metallicity tables.
    """
    def linearinterpolate(x, other):
        newBCs = ((other[x.name] - x)/(BC2.FeH-BC1.FeH))*(FeH-BC1.FeH) + x
        return newBCs

    AvRvKey = f"Av={Av:0.1f}:Rv={Rv:0.1f}"
    tab1 = BC1.data[AvRvKey]
    tab2 = BC2.data[AvRvKey]

    interpolated = tab1.apply(lambda x: linearinterpolate(x, tab2))
    return interpolated

def get_mags(Teff, logg, logL, BCTab):
    return _get_mags(Teff, logg, logL, BCTab)

