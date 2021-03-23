
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Genetic Methods for Neural Nets Training for Trading                                       -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/GeneticTraining                                        -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis as m_kurtosis
from scipy.stats import skew as m_skew

# ---------------------------------------------------------------------------- FEATURES BASIC STATISTICS -- #
# --------------------------------------------------------------------------------------------------------- #

def data_profile(p_data, p_type, p_mult):
    """
    OHLC Prices Profiling (Inspired in the pandas-profiling existing library)

    Parameters
    ----------

    p_data: pd.DataFrame
        A data frame with columns of data to be processed

    p_type: str
        indication of the data type: 
            'ohlc': dataframe with TimeStamp-Open-High-Low-Close columns names
            'ts': dataframe with unknown quantity, meaning and name of the columns
    
    p_mult: int
        multiplier to re-express calculation with prices,
        from 100 to 10000 in forex, units multiplication in cryptos, 1 for fiat money based assets
        p_mult = 10000

    Return
    ------

    r_data_profile: dict
        {}
    
    References
    ----------

    https://github.com/pandas-profiling/pandas-profiling

    """

    # copy of input data
    f_data = p_data.copy()

    # interquantile range
    def f_iqr(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        return  q1 - q3
    
    # outliers function (returns how many were detected, not which ones or indexes)
    def f_out(param_data):
        q1 = np.percentile(param_data, 75, interpolation = 'midpoint')
        q3 = np.percentile(param_data, 25, interpolation = 'midpoint')
        lower_out = len(np.where(param_data < q1 - 1.5*f_iqr(param_data))[0])
        upper_out = len(np.where(param_data > q3 + 1.5*f_iqr(param_data))[0])
        return [lower_out, upper_out]


    # -- OHLCV PROFILING -- #
    if p_type == 'ohlc':

        # initial data
        ohlc_data = p_data[['open', 'high', 'low', 'close', 'volume']].copy()

        # data calculations
        ohlc_data['co'] = round((ohlc_data['close'] - ohlc_data['open'])*p_mult, 2)
        ohlc_data['hl'] = round((ohlc_data['high'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ol'] = round((ohlc_data['open'] - ohlc_data['low'])*p_mult, 2)
        ohlc_data['ho'] = round((ohlc_data['high'] - ohlc_data['open'])*p_mult, 2)

        # original data + co, hl, ol, ho columns
        f_data = ohlc_data.copy()
    
    # basic data description
    data_des = f_data.describe(percentiles=[0.25, 0.50, 0.75, 0.90])

    # add skewness metric
    skews = pd.DataFrame(m_skew(f_data)).T
    skews.columns = list(f_data.columns)
    data_des = data_des.append(skews, ignore_index=False)

    # add kurtosis metric
    kurts = pd.DataFrame(m_kurtosis(f_data)).T
    kurts.columns = list(f_data.columns)
    data_des = data_des.append(kurts, ignore_index=False)
    
    # add outliers count
    outliers = [f_out(param_data=f_data[col]) for col in list(f_data.columns)]
    negative_series = pd.Series([i[0] for i in outliers], index = data_des.columns)
    positive_series = pd.Series([i[1] for i in outliers], index = data_des.columns)
    data_des = data_des.append(negative_series, ignore_index=True)
    data_des = data_des.append(positive_series, ignore_index=True)
    
    # index names
    data_des.index = ['count', 'mean', 'std', 'min', 'q1', 'median', 'q3', 'p90',
                      'max', 'skew', 'kurt', 'n_out', 'p_out']

    return np.round(data_des, 2)


# --------------------------------------------------------------------------------- FEATURES BASIC PLOTS -- #
# --------------------------------------------------------------------------------------------------------- #

def visual_profile(p_data, p_type, p_mult):

    

    return 1
