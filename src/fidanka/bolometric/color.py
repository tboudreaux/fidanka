from fidanka.bolometric.load import load_bol_table

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

REJECTNAMES = ["Teff", "logg", "[Fe/H]", "Av", "Rv"]


def get_interpolated_FeHTable(BC1, BC2, Av, Rv, FeH):
    """
    Get the bolometric correction table for a given metallicity, Av, and Rv. If
    that mettallicity is not in the BC tables, linearlly interpolate between
    the two closest metallicity tables.
    """

    def linearinterpolate(x, other):
        newBCs = ((other[x.name] - x) / (BC2.FeH - BC1.FeH)) * (FeH - BC1.FeH) + x
        return newBCs

    AvRvKey = f"Av={Av:0.1f}:Rv={Rv:0.1f}"
    tab1 = BC1.data[AvRvKey]
    tab2 = BC2.data[AvRvKey]

    interpolated = tab1.apply(lambda x: linearinterpolate(x, tab2))
    return interpolated
