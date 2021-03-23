
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

import datetime
import pandas as pd
import ccxt
import time
exchanges = print(ccxt.exchanges)

# -------------------------------------------------------------------------------- Get Historical Prices -- #

# References
# https://github.com/ccxt/ccxt/wiki/Manual#public-api
# from variable id

s1 = 'd6fudVgjqEfhJHE7Iw2TAh9NvkCWRkTddMlwzSV9YVeqjWdCGDU7cuMuQsupUoHS'
s2 = 'xXszWxXy8irosdRjEa9Wa0ae7NEFP8l8NST3q4PV59Qdk6Mti7w07MpexTZeNfNJ'

exchange_id = 'binance'
exchange_class = getattr(ccxt, exchange_id)
exchange = exchange_class({
    'apiKey': s1,
    'secret': s2,
    'timeout': 30000,
    'enableRateLimit': False,
})

markets = exchange.load_markets()
# etheur1 = exchange.markets['ETH/EUR']      # get market structure by symbol
# ss = exchange.has['fetchOHLCV']

# -- masive OHLCV data -- 
p_ini_date = '2021-01-01 00:00:00'
p_end_date = '2021-03-23 00:00:00'
p_asset = 'ETH/USDT'
p_freq = '1m'

def massive_ohlcv(p_ini_date, p_end_date, p_asset):
    """
    """

    # initial date
    ini_date = int(datetime.datetime.strptime(p_ini_date, '%Y-%m-%d %H:%M:%S').timestamp())*1000
    # final date
    end_date = int(datetime.datetime.strptime(p_end_date, '%Y-%m-%d %H:%M:%S').timestamp())*1000
    # first call
    prices = []
    prices.append(exchange.fetch_ohlcv (p_asset, p_freq, ini_date))

    # number of batches (based on limit by exchange and timeframe)
    # batches = int((end_date - ini_date)/1000/60/60/exchange.rateLimit)+1
    batches = 2

    # instrument
    instrument = 'ETH/USDT'

    for itera in range(0, batches):
        time.sleep(exchange.rateLimit*5/1000)
        print('iteration: ', itera)
        prices.append(exchange.fetch_ohlcv(instrument, p_freq, prices[itera][-1][0]))

    df = pd.concat([pd.DataFrame(data_0) for data_0 in prices])
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df['timestamp'] = pd.to_datetime(df['timestamp']*1000000)
    df.reset_index(inplace=True, drop=True)

    return df


df_prices = massive_ohlcv(p_ini_date, p_end_date, p_asset)
