import re
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def estimate_RMS_function(data):
    filters = data.filter(regex=r'F\d{3}W')
    filterNames = filters.filter(regex=r"F\d{3}W$").columns
    RMSFuncs = dict()
    for filterName in filterNames:
        nominalPhotometry = filters[filterName].values
        RMSPhotometry = filters[f"{filterName}_RMS"].values
        exposuresMeasured = filters[f"{filterName}_NG"].values
        goodPhotometry = nominalPhotometry[exposuresMeasured > 1]
        goodRMS = RMSPhotometry[exposuresMeasured > 1]
        goodRMS = goodRMS[np.argsort(goodPhotometry)]
        goodPhotometry = goodPhotometry[np.argsort(goodPhotometry)]
        print(goodRMS.shape, goodPhotometry.shape)
        RMSFunc = interp1d(goodPhotometry, goodRMS, kind='linear', fill_value=0.01, bounds_error=False)
        RMSFuncs[filterName] = RMSFunc
    return RMSFuncs
