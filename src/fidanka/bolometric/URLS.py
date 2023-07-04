lookup_table = {
    "cfht": "https://waps.cfa.harvard.edu/MIST/BC_tables/CFHTugriz.txz",
    "megacam": "https://waps.cfa.harvard.edu/MIST/BC_tables/CFHTugriz.txz",
    "decam": "https://waps.cfa.harvard.edu/MIST/BC_tables/DECam.txz",
    "galxex": "https://waps.cfa.harvard.edu/MIST/BC_tables/GALEX.txz",
    "hrc": "https://waps.cfa.harvard.edu/MIST/BC_tables/HST_ACSHR.txz",
    "wfc": "https://waps.cfa.harvard.edu/MIST/BC_tables/HST_ACSWF.txz",
    "wfc3": "https://waps.cfa.harvard.edu/MIST/BC_tables/HST_WFC3.txz",
    "wfpc2": "https://waps.cfa.harvard.edu/MIST/BC_tables/HST_WFPC2.txz",
    "jwst": "https://waps.cfa.harvard.edu/MIST/BC_tables/JWST.txz",
    "lsst": "https://waps.cfa.harvard.edu/MIST/BC_tables/LSST.txz",
    "panstars": "https://waps.cfa.harvard.edu/MIST/BC_tables/PanSTARRS.txz",
    "sdss": "https://waps.cfa.harvard.edu/MIST/BC_tables/SDSSugriz.txz",
    "skymapper": "https://waps.cfa.harvard.edu/MIST/BC_tables/SkyMapper.txz",
    "subaru hyper suprime-cam": "https://waps.cfa.harvard.edu/MIST/BC_tables/HSC.txz",
    "subaru hyper": "https://waps.cfa.harvard.edu/MIST/BC_tables/HSC.txz",
    "suprime-cam": "https://waps.cfa.harvard.edu/MIST/BC_tables/HSC.txz",
    "int": "https://waps.cfa.harvard.edu/MIST/BC_tables/IPHAS.txz",
    "spitzer irac": "https://waps.cfa.harvard.edu/MIST/BC_tables/SPITZER.txz",
    "s-plus": "https://waps.cfa.harvard.edu/MIST/BC_tables/SPLUS.txz",
    "swift": "https://waps.cfa.harvard.edu/MIST/BC_tables/Swift.txz",
    "ubv(ri)c + 2mass + kepler + hipparcos + tycho + gaia": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "ubv(ri)c": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "2mass": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "kepler": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "hipparcos": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "tycho": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "gaia": "https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz",
    "ukiiss": "https://waps.cfa.harvard.edu/MIST/BC_tables/UKIDSS.txz",
    "uvit": "https://waps.cfa.harvard.edu/MIST/BC_tables/UVIT.txz",
    "vista": "https://waps.cfa.harvard.edu/MIST/BC_tables/VISTA.txz",
    "washington + stromgren + ddo51": "https://waps.cfa.harvard.edu/MIST/BC_tables/WashDDOuvby.txz",
    "washington": "https://waps.cfa.harvard.edu/MIST/BC_tables/WashDDOuvby.txz",
    "stromgren": "https://waps.cfa.harvard.edu/MIST/BC_tables/WashDDOuvby.txz",
    "ddo51": "https://waps.cfa.harvard.edu/MIST/BC_tables/WashDDOuvby.txz",
    "wfirst": "https://waps.cfa.harvard.edu/MIST/BC_tables/WFIRST.txz",
    "wise": "https://waps.cfa.harvard.edu/MIST/BC_tables/WISE.txz",
}


def get_valid_bol_table_names():
    return list(lookup_table.keys())


def get_valid_bol_table_URLS():
    return [x[1] for x in lookup_table.items()]


# # TODO: Automate the makeing of this checksum dict so that the contents of a download mist table
# #       may be automatically checked if they are okay.
# checksums ={
#         'JWST':
#         {
#             'fehm0': 'sdfjhasdfjasf',
#             ...
#             },
#         ...
#         }
