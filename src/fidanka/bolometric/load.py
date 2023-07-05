from fidanka.bolometric.URLS import lookup_table
from fidanka.misc.utils import get_logger
from fidanka.bolometric import ext_data

import importlib.resources as pkg
import hashlib

import re
import pandas as pd

import numpy as np

from io import StringIO
import os

from typing import List, Dict, Union, Tuple

import requests
from tqdm import tqdm
import pickle as pkl

import tarfile
import warnings

HEADER_MATCH = re.compile(r"(#\s+1.+\n#\s+Teff.+)")


class _endCorrector:
    """
    Helper class to correct the end index of the file pointer for the last
    sub-table
    """

    def __init__(self, length: int):
        self.length = length

    def start(self) -> int:
        return self.length

    def end(self) -> int:
        return self.length


def load_bol_table(filename: str) -> Dict[str, pd.DataFrame]:
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
    with open(filename, "r") as bolTable:
        contents = bolTable.read()
    headers = list(re.finditer(HEADER_MATCH, contents))
    tableMeta = [load_sub_table_metadata(filename, x.start()) for x in headers]

    startPoints = [x.end() for x in headers]
    endPoints = [x.start() for x in [*headers[1:], _endCorrector(len(contents))]]

    tables = [
        load_sub_bol_table(contents[start:end], m[1])
        for start, end, m in zip(startPoints, endPoints, tableMeta)
    ]
    AvRv = [table[["Av", "Rv"]].iloc[0] for table in tables]
    # identString = [f"Av={avrv['Av']}:Rv={avrv['Rv']}" for avrv in AvRv]
    out = {(avrv["Av"], avrv["Rv"]): table for avrv, table in zip(AvRv, tables)}
    return out


def load_sub_bol_table(contents: str, names: list) -> pd.DataFrame:
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


def load_bol_table_metadata(filename: str) -> Dict[str, Union[str, int]]:
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
    with open(filename, "r") as bolTable:
        headerLine = bolTable.readline()[1:]
        keys = bolTable.readline()[1:]
        values = bolTable.readline()[1:]
        _ = bolTable.readline()  # read the blank line and dont save it
        description = headerLine.rstrip().lstrip()
        values = [int(x) for x in re.findall(r"\d+", values)]
        numKeys = len(values)
        keys = [keys[i * 8 + 1 : i * 8 + 9].rstrip().lstrip() for i in range(numKeys)]

        filterNums = bolTable.readline()[1:]
        colNames = bolTable.readline()[1:]
    filterNums = [int(x) for x in re.findall(r"\d+", filterNums)]
    colNames = colNames.split()
    out = {
        "desc": description,
        "sm1": {k: v for k, v in zip(keys, values)},
    }
    return out


def load_sub_table_metadata(filename: str, pointer: int) -> Tuple[int, List[str]]:
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
    with open(filename, "r") as bolTable:
        bolTable.seek(pointer)
        filterNums = bolTable.readline()[1:]
        colNames = bolTable.readline()[1:]
    filterNums = [int(x) for x in re.findall(r"\d+", filterNums)]
    colNames = colNames.split()
    return len(filterNums), colNames


def download_MIST_bol_table(ID: str, folder=None) -> Tuple[str, str]:
    """
    Download bolometric correction tables from the MIST
    data-products site. Valid keys can be checked with
    fidanka.bolometric.URLS.get_valid_bol_table_names().

    By default files will be downloaded to ~/.fidanka/bol/MIST/
    however, this can be changed using the folder keyword argument.
    Note, this function will not extract or verify downloads.

    If the file already exists then nothing will be downloaded.

    Parameters
    ----------
        ID : str
            Name of bol correction table to fetch. Valid names can be
            checked using the fidanka.bolometric.URLS.get_valid_bol_table_names()
            function
        folder : Union[str, None], default=None
            The folder to save bolometric correction tables too. By default
            this will be ~/.fidanka/bol/MIST/

    Returns
    -------
        filePath : str
            The path to the downloaded file
        folder : str
            the folder where the file was saved to (same as the input
            parameter folder)

    Raises
    ------
        ValueError
            If ID is not a key in the lookup_table.


    Examples
    --------
    If you want to download the JWST bolometric correction tables to
    the default directory

    >>> download_MIST_bol_table("JWST")

    then if you do

    >>> ls ~/.fidanka/bol/MIST/
    >>> Out[1]: JWST.tar.gz


    """
    logger = get_logger("fidanka.bolometric.load.download_MIST_bol_table")
    if not folder:
        folder = os.path.join(os.path.expanduser("~"), ".fidanka", "bol", "MIST")
    if not os.path.exists(folder):
        os.makedirs(folder)

    url = lookup_table.get(ID.lower())
    if not url:
        validKeys = ", ".join(lookup_table.keys())
        raise ValueError(f"Unknown MIST Bol Table ID: {ID}. Valid keys are {validKeys}")
    filename = url.split("/")[-1]
    filePath = os.path.join(folder, filename)
    if os.path.exists(filePath):
        logger.info(f"File already exists at {filePath}")
        return filePath, folder

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    logger.info(
        f"Downloading bolometric correction archive of size {total_size_in_bytes/1000} kB"
    )
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc=f"Fetching {ID} bolometric correction tables",
    )
    with open(filePath, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    return filePath, folder


def verify_MIST_bol_tables(paths: List[str]) -> bool:
    """
    Verify extracted files from the MIST bolometric correction
    tables (based on their versions July 4th 2023). If all
    of the files given in paths match the checksums stashed in
    the fidanka package data then this function will return true
    otherwise this will return false.

    Parameters
    ----------
        paths : List[str]
            List of paths to files. Each path must have a basename which exists within
            the stashed checksum file. This should include every bolometric correction
            table avalible from MIST as of July 4th 2023.

    Returns
    -------
        okay : bool
            flag defining paths as either verified or not. If all files in the paths
            are both in the stashed checksums and the MD5 checksums of all files
            in paths match, then this will be true. In all other senarios this will
            return as false

    Warns
    -----
        - If a file in paths is not found in the stashed checksums then a warning will
          be raised.
    """
    logger = get_logger("fidanka.bolometric.load.verify_MIST_bol_table")
    logger.info("Loading stashed checksums...")
    with pkg.path(ext_data, "checksums") as checksumPath:
        with open(checksumPath, "r") as f:
            checksums = f.read().split("\n")
    checksums = [x for x in checksums if len(x.lstrip()) > 0]
    checksums = [x for x in checksums if x.lstrip()[0] != "#"]
    logger.info("Stashed checksums loaded!")
    okay = True
    logger.info("Checking paths against stashed checksums...")
    for path in paths:
        file = os.path.basename(path)
        line = [x for x in checksums if file in x]
        if len(line) != 1:
            logger.warn(f"File {path} not identified in checksum stash!")
            return False
        checksum = line[0].split(":")[1]
        with open(path, "rb") as f:
            hexdigest = hashlib.md5(f.read()).hexdigest()
        okay &= hexdigest == checksum
        if hexdigest != checksum:
            logger.warn(
                f"File {path} checksum {hexdigest} does not match stashed checksum {checksum}!"
            )
    return okay


def fetch_MIST_bol_table(ID: str, folder=None) -> Tuple[str, str]:
    """
    Fetch (download, extract, and verify) bolometric correction tables for a given
    MIST table ID (valid names can be checked with
    fidanka.bolometric.URLS.get_valid_bol_table_names())

    First, a check will be done to check if files for a given ID have already
    been downloaded and extracted to <folder>. If so they will have their checksums
    validated. If all checks passed then the function will stop here.

    If some of the checks fail then the tarball will be redownloaded from
    the MIST website, these tarballs will then be extracted to some folder
    ~/.fidanka/bol/MIST/<ID> (or more generally <folder>/<ID>).

    Parameters
    ----------
        ID : str
            Name of bol correction table to fetch. Valid names can be
            checked using the fidanka.bolometric.URLS.get_valid_bol_table_names()
            function
        folder : Union[str, None], default=None
            The folder to save bolometric correction tables too. By default
            this will be ~/.fidanka/bol/MIST/

    Returns
    -------
        ID : str
            The MIST id requested
        sfp : str
            Path to the "subfolder" where files are extracted too. This will
            be <folder>/<ID>. By default then ~/.fidanka/bol/MIST/<ID>

    Warnings
    --------
        - If the bolometric correction tables checksums cannot be validated after
          being downloaded and extracted

    Examples
    --------
    If you want to download the JWST bolometric correction tables to
    the default directory

    >>> fetch_MIST_bol_table("JWST")

    then if you do

    >>> ls ~/.fidanka/bol/MIST/JWST

    you should see all of the JWST bolometric correction tables, named like
    feh[m|p][0-9]{3}.JWST
    """
    logger = get_logger("fidanka.bolometric.load.fetch_MIST_bol_table")
    path, folder = download_MIST_bol_table(ID, folder=folder)
    subFolder = os.path.join(os.path.basename(path).split(".")[0])
    sfp = os.path.join(folder, subFolder)
    if not os.path.exists(sfp):
        os.makedirs(sfp)
    else:
        logger.info(
            "Pre-existing bolometric correction tables detected, verifying checksums"
        )
        paths = os.listdir(sfp)
        okay = verify_MIST_bol_tables(paths)
        if okay:
            return ID, sfp
    logger.info(f"Extracting File at {path}")
    with tarfile.open(path) as tarball:
        tarball.extractall(path=sfp)
    paths = os.listdir(sfp)
    okay = verify_MIST_bol_tables(paths)
    if not okay:
        logger.warn("Bolometric Tables do not match stashed checksums")
        warnings.warn(
            "Fetched Bolometric Correction tables do not match stashed checksums!"
        )

    return ID, sfp


def get_MIST_paths_FeH(paths: List[str]):
    tableFeHs = np.empty(len(paths))
    for idx, path in enumerate(paths):
        tabFeH = re.search(r"feh([mp]\d+)", path).group(1)
        if tabFeH[0] == "m":
            sign = -1
        else:
            sign = 1
        tabFehNS = tabFeH[1:]
        decimal = sign * float(f"{tabFehNS[0]}.{tabFehNS[1:]}")
        tableFeHs[idx] = decimal
    return tableFeHs


if __name__ == "__main__":
    files = os.listdir("/Users/tboudreaux/.fidanka/bol/MIST/JWST/")
    paths = [
        os.path.join("/Users/tboudreaux/.fidanka/bol/MIST/JWST/", f) for f in files
    ]
    print(verify_MIST_bol_tables(paths))
