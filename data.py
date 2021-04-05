
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Applications of Genetic Methods for Feature Engineering and Hyperparameter Optimization    -- #
# -- -------- for Neural Networks.                                                                       -- #
# -- script: data.py : python script for input/output data functions                                     -- #
# -- author: IFFranciscoME - if.francisco.me@gmail.com                                                   -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: https://github.com/IFFranciscoME/GeneticMethods                                         -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import os
import datetime
import pandas as pd
import ccxt
import time


# --------------------------------------------------------------------------- INSTANTIATE BINANCE CLIENT -- #
# --------------------------------------------------------------------------- -------------------------- -- #

def ini_binance(p_s1, p_s2):
    """
    Function to instantiate a exchange_class from ccxt library, connecting to binance public API

    Parameters
    ----------
    
    p_s1: str
        Token from Binance (Store it in environment variable for safety)

        p_s1 = 'Ax13HNjx ... '
        
    p_s2: Token from Binance (Store it in environment variable for safety)

        p_s2 = 'Ax13HNjx ... '

    Returns
    -------

    exchange: exchange_class 
        object returned from ccxt

    References
    ----------
    https://github.com/ccxt/ccxt


    """

    # selected exchange (one of the major in volume)
    exchange_id = 'binance'  

    # instantiate class for particularly selected exchange
    exchange_class = getattr(ccxt, exchange_id)
    
    # parameters for the call 
    parameters = {'apiKey': p_s1, 'secret': p_s2, 'timeout': 30000,
                  'enableRateLimit': True, 'rateLimit': 10000}

    # instantiate class
    exchange = exchange_class(parameters)

    # get markets available in exchange
    markets = exchange.load_markets()

    # check if exchange supports fetch OHLCV historical data
    fetch_info = exchange.has['fetchOHLCV']

    # result
    d_exchange = {'markets': markets, 'fetch_info': fetch_info}

    return d_exchange


# ---------------------------------------------------------- DOWNLOADER OF MASSIVE HISTORICAL OHLCV DATA -- #
# --------------------------------------------------------------------------------------------------------- #

def massive_ohlcv(p_class, p_ini_date, p_end_date, p_asset, p_freq, p_verbose):
    """
    To fetch massive cryptoassets prices in the format of OHLCV, by using ccxt public api-s for 
    different exchanges. Some considerations must be taken for each exchange, for example, the maximum
    calls and/or returned historical info per unit of time.

    Parameters
    ----------
    
    p_class: ccxt.exchange_class
        Instantiated class using ccxt exchange_class
        
        exchange = exchange_class({'apiKey': s1, 'secret': s2, 'timeout': 30000, 'enableRateLimit': False})

    p_ini_date: str
        initial datetime in the format : 'YYYY-MM-DD HH:MM:SS' or other supported format for datetime

        p_ini_date = '2021-03-21 18:00:00'       

    p_end_date: str
        final datetime in the format : 'YYYY-MM-DD HH:MM:SS' or other supported format for datetime

        p_end_date = '2021-03-22 10:00:00'

    p_asset: str
        Name of the crypto asset according to exchange_class.markets

        p_asset = 'ETH/USDT'

    p_freq: str
        frequency of prices, supported by this function are: 
            '1m': 1 minute, '5m': 5 minutes, '15m': 15 minutes, '30m': 30 minutes,
            '1h': 1 hour, '2h': 2 hours, '4h': 4 hours, '8h': 8 hours, '12h': 12 hours,
            '1d': daily, '1w': weekly

        p_freq = '1m'

    p_verbose: bool
        Whether to print progress indication message

        p_verbose = True
    
    Returns
    -------
        df_prices: pd.DataFrame
            With pandas dataframe with a numerical index and columns: timestap, open, high, low, close, volume
            *timestamp column will be returned with UTC timezone.
    
    References
    ----------
    https://github.com/ccxt/ccxt/wiki/Manual#public-api

    
    **** Warning ****
    date will be automatically generated with your timezone diff from UTC therefore consider
    specify a ini_date adjusting from your tz to UTC, 
    e.g. manually specify a datetime 6 Hours less than intender if GMT-6 is your time zone.
    
    """

    # dates in milisecond format
    ini_date = int(datetime.datetime.strptime(p_ini_date, '%Y-%m-%d %H:%M:%S').timestamp())*1000
    end_date = int(datetime.datetime.strptime(p_end_date, '%Y-%m-%d %H:%M:%S').timestamp())*1000
    # first call
    prices = []
    prices.append(p_class.fetch_ohlcv (p_asset, p_freq, ini_date))

    # period info for further lazy calculations
    d_periods = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600,
                 '2h': 7200, '4h': 14400, '8h': 28800, '12h': 43200, '1d': 86400,
                 '1w': 604800}

    # number of batches (based on limit by exchange and timeframe) = example case with 1m
    batches = int((end_date - ini_date)/1000/d_periods[p_freq]/p_class.rateLimit) + 1

    # iterations with delay to fetch historical OHLCV candles from selected exchange
    for itera in range(0, batches+1):

        # messages print in case of p_verbose=True
        if p_verbose:
            print('iteration: ', itera, ' of: ', batches)

        # time delay according to exchange based restriction for calls
        time.sleep(p_class.rateLimit*5/1000)

        # define next timestamp from where to begin, by using last timestamp of the recent call
        next_time = prices[itera][-1][0] + d_periods[p_freq]*1000

        # append to a list
        prices.append(p_class.fetch_ohlcv(p_asset, p_freq, next_time))

    # format to dataframe
    df_data = pd.concat([pd.DataFrame(data) for data in prices])
    # df_data.reset_index(inplace=True, drop=True)
    df_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    # df_data['timestamp'] = pd.to_datetime(df_data['timestamp']*1000000)
    df_data.index = pd.to_datetime(df_data['timestamp']*1000000)
    df_data.drop('timestamp', inplace=True, axis=1)

    file_name = p_asset.replace('/', '_') + '_' + p_freq + '.csv'

    df_data.to_csv('files/' + file_name, index_label='timestamp')

    return df_data


# ----------------------------------------------------------------------------------- masive OHLCV data -- #

# s1 = os.environ['K1']
# s2 = os.environ['K2']
# p_class = ini_binance(p_s1=s1, p_s2=s2)
# p_ini_date = '2018-01-01 00:00:00'
# p_end_date = '2021-03-22 23:59:00'
# p_asset = 'ETH/USDT'
# p_freq = '1m'
# p_verbose = True

# -- RUN ONLY IF YOU WANT TO FETCH A LARGE HISTORICAL DATA
# df_prices = massive_ohlcv(p_class, p_ini_date, p_end_date, p_asset, p_freq, p_verbose)

# ------------------------------------------------------------------------------- Read Masive OHLCV data -- #

# -- RUN IF YOU WANT TO READ SAVED CSV FILE
df_prices = pd.read_csv('files/prices/ETH_USDT_8h.csv')
