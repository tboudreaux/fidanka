import re
import pandas as pd

from io import StringIO

from typing import List, Dict, Union, Tuple

HEADER_MATCH = re.compile(r'(#\s+1.+\n#\s+Teff.+)')

class _endCorrector:
    """
    Helper class to correct the end index of the file pointer for the last
    sub-table
    """
    def __init__(self, length):
        self.length = length
    def start(self):
        return self.length
    def end(self):
        return self.length

def load_bol_table(
        filename: str
        ) -> Dict[str, pd.DataFrame]:
    """
    Load a MIST formated bolometric correction table into a dictionary
    of pandas DataFrames. The dictionary will be keyed by the Av and Rv.

    Parameters
    ----------
    filename : str
        Path to the bolometric correction table

    Returns
    -------
    out : Dict[str, pd.DataFrame]
        Dictionary of pandas DataFrames keyed by Av and Rv.
    """
    with open(filename, 'r') as bolTable:
        contents = bolTable.read()
    headers = list(re.finditer(HEADER_MATCH,contents))
    tableMeta = [load_sub_table_metadata(filename, x.start()) for x in headers]

    startPoints = [x.end() for x in headers]
    endPoints = [x.start() for x in [*headers[1:], _endCorrector(len(contents))]]

    tables = [load_sub_bol_table(contents[start:end], m[1])
              for start,end,m in zip(
                  startPoints,
                  endPoints,
                  tableMeta
                  )
              ]
    AvRv = [table[["Av","Rv"]].iloc[0] for table in tables]
    # identString = [f"Av={avrv['Av']}:Rv={avrv['Rv']}" for avrv in AvRv]
    out = {(avrv['Av'], avrv['Rv']): table for avrv, table in zip(AvRv, tables)}
    return out

def load_sub_bol_table(
        contents: str,
        names:list
        ) -> pd.DataFrame:
    """
    Load a single sub-table from a MIST formated bolometric correction table.

    Parameters
    ----------
    contents : str
        Contents of the sub-table

    Returns
    -------
    table : pd.DataFrame
        Pandas DataFrame containing the sub-table
    """
    table = pd.read_csv(StringIO(contents), names=names, delimiter=r"\s+")
    assert isinstance(table, pd.DataFrame)
    return table


def load_bol_table_metadata(
        filename: str
        ) -> Dict[str, Union[str, int]]:
    """
    Load the metadata from a MIST formated bolometric correction table.
    Do this efficiently by only reading the first few lines of the file.

    Parameters
    ----------
    filename : str
        Path to the bolometric correction table

    Returns
    -------
    out : Dict[str, Union[str, int]]
        Dictionary containing the metadata
    """
    with open(filename, 'r') as bolTable:
        headerLine = bolTable.readline()[1:]
        keys = bolTable.readline()[1:]
        values = bolTable.readline()[1:]
        _ = bolTable.readline() # read the blank line and dont save it
        description = headerLine.rstrip().lstrip()
        values = [int(x) for x in re.findall(r'\d+', values)]
        numKeys = len(values)
        keys = [keys[i*8+1:i*8+9].rstrip().lstrip() for i in range(numKeys)]

        filterNums = bolTable.readline()[1:]
        colNames = bolTable.readline()[1:]
    filterNums = [int(x) for x in re.findall(r'\d+', filterNums)]
    colNames = colNames.split()
    out = {
            'desc': description,
            'sm1': {k: v for k, v in zip(keys, values)},
            }
    return out

def load_sub_table_metadata(
        filename: str,
        pointer:int
        ) -> Tuple[int, List[str]]:
    """
    Load the metadata from a single sub-table in a MIST formated bolometric
    correction table.

    Parameters
    ----------
    filename : str
        Path to the bolometric correction table
    pointer : int
        Pointer to the start of the sub-table

    Returns
    -------
    out : Tuple[int, List[str]]
        Tuple containing the number of filters and the column names
    """
    with open(filename, 'r') as bolTable:
        bolTable.seek(pointer)
        filterNums = bolTable.readline()[1:]
        colNames = bolTable.readline()[1:]
    filterNums = [int(x) for x in re.findall(r'\d+', filterNums)]
    colNames = colNames.split()
    return len(filterNums), colNames
