"""
Created on Fri Apr  9 14:40:43 2021

@author: omkar
"""


from alpaca_trade_api.rest import REST

import alpaca_trade_api as tradeapi


api = REST('PKTEIXKKZSEUW0APR22K', 'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',base_url='https://paper-api.alpaca.markets',)


#df=api.polygon.historic_agg_v2("GOOGL",1, tradeapi.rest.TimeFrame.Hour, "2020-02-04", "2021-02-08")

df=api.get_bars("AAPL", tradeapi.rest.TimeFrame.Minute, "2021-01-06", "2021-02-06", adjustment='raw').df

print(df)
#api = tradeapi.REST()

""" Get daily price data for AAPL over the last 5 trading days.
api.get_bars("AAPL", t, "2021-02-08", "2021-02-08", limit=10, adjustment='raw').df
df=api.get_bars("AAPL", t, "2021-01-06", "2021-02-06", adjustment='raw').df

df=api.polygon.historic_agg_v2('AAPL', 1, 'day',_from='2020-02-01',to='2016-02-01').df

print(df)"""