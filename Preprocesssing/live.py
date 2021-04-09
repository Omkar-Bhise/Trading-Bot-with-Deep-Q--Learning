# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:40:43 2021

@author: omkar
"""

from alpaca_trade_api.stream import Stream

async def trade_callback(t):
    print('trade', t)

async def bar_callback(b):
    print('bar', b)

async def quote_callback(q):
    print('quote', q)


# Initiate Class Instance
stream = Stream('PKTEIXKKZSEUW0APR22K',
                'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',
                base_url='https://paper-api.alpaca.markets',
                data_feed='iex')  # <- replace to SIP if you have PRO subscription

# subscribing to event
stream.subscribe_bars(bar_callback, 'AAPL')
stream.subscribe_bars(bar_callback, 'TSLA')
#stream.subscribe_quotes(quote_callback, 'IBM')

stream.run()
