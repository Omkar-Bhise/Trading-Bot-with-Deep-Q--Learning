from alpaca_trade_api.rest import REST

import alpaca_trade_api as tradeapi


def load_stock_data():
    api = REST('PKTEIXKKZSEUW0APR22K', 'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',
               base_url='https://paper-api.alpaca.markets', )

    # df=api.polygon.historic_agg_v2("GOOGL",1, tradeapi.rest.TimeFrame.Hour, "2020-02-04", "2021-02-08")

    data = api.get_bars("TSLA", tradeapi.rest.TimeFrame.Day, "2016-04-23", "2020-03-01", adjustment='raw').df
    data.index.rename("Date")

    return (data)


def test_stock_data():
    api = REST('PKTEIXKKZSEUW0APR22K', 'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',
               base_url='https://paper-api.alpaca.markets', )

    # df=api.polygon.historic_agg_v2("GOOGL",1, tradeapi.rest.TimeFrame.Hour, "2020-02-04", "2021-02-08")

    data = api.get_bars("TSLA", tradeapi.rest.TimeFrame.Day, "2020-03-02", "2021-04-23", adjustment='raw').df
    data.index.rename("Date")

    return (data)
