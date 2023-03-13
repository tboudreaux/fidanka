import re
import pandas as pd
from io import StringIO
from itertools import islice

from typing import Any

def read_iso(
        filename : str
        ) -> dict[float, pd.DataFrame]:
    """
    Read in a MIST formated isochrone to a list of DataFrame

    Parameters
    ----------
        filename : str
            filepath pointing to the isochrone file

    Returns
    -------
        isos : dict[float, pd.DataFrame]
            Dictionary of isochrones loaded and indexed by their ages in 
            Gyr
    """
    with open(filename) as f:
        contents = f.read()

    lines = contents.split('\n')

    tabSep = r'((?:# number of EEPs, cols =\s+)(\d+)(?:\s+)(\d+))'
    tables = map(lambda x: (x[0], re.search(tabSep, x[1])), enumerate(lines))
    fTables = list(filter(lambda x: x[1], tables))

    sepTables = map(lambda x: lines[x[0][0]:x[1][0]], zip(fTables[:-1], fTables[1:]))

    isos = dict()
    for table in sepTables:
        header = re.findall(r'[^\s\\#]+', table[2])
        okayTab = filter(lambda x: x!='' and x.lstrip()[0] != '#', table)
        strTab = '\n'.join(okayTab)
        df = pd.read_csv(StringIO(strTab), sep=r'\s+', names=header)

        assert isinstance(df, pd.DataFrame)
        age = 10**df['log10_isochrone_age_yr'].iloc[0]
        isos[age] = df
    return isos

def read_iso_metadata(
        filename : str
        ) -> dict[str, Any]:
    """
    Efficiently read in isochrone metadata without opening entire file

    Parameters
    ----------
        filename : str
            filepath pointing to the isochrone file.

    Returns
    -------
        metadata : dict[str, Any]
            Dictionary of metadata relating to isochrone pulled from the
            first 7 rows of the isochrone file.
    """
    with open(filename) as f:
        rawHeader = list(islice(f,7))

    MESAVersion = rawHeader[0].split('=')[1].rstrip().lstrip()
    MESARevision = rawHeader[1].split('=')[1].rstrip().lstrip()
    numRow = [float(x) for x in rawHeader[4][1:].rstrip().lstrip().split()]

    metadata = {
            'MESAVersion': MESAVersion,
            'MESARevision': MESARevision,
            'Yinit': numRow[0],
            'Zinit': numRow[1],
            '[Fe/H]': numRow[2],
            '[a/Fe]': numRow[3],
            'v/vcrit': numRow[4]
            }
    return metadata

