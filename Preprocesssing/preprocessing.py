"""
Created on Fri Apr  9 14:40:43 2021

@author: omkar
"""


from alpaca_trade_api.rest import REST
import pandas as pd
import alpaca_trade_api as tradeapi


def load_train_data():
    api = REST('PKTEIXKKZSEUW0APR22K', 'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',
               base_url='https://paper-api.alpaca.markets', )

    # df=api.polygon.historic_agg_v2("GOOGL",1, tradeapi.rest.TimeFrame.Hour, "2020-02-04", "2021-02-08")

    data = api.get_bars("TSLA", tradeapi.rest.TimeFrame.Day, "2016-04-23", "2020-03-01", adjustment='raw').df
    data.index.rename("Date")
    strt_date = datetime.strptime("2016-04-23", "%Y-%m-%d")
    end_date = datetime.strptime("2020-03-01", "%Y-%m-%d")
    nlp_df = pd.read_csv("sentiment.csv")
    data["sentiment"] = nlp_df[nlp_df["Datetime"]>strt_date & nlp_df["Datetime"]<end_date]["Tesla"]
    return (data)


def load_test_data():
    api = REST('PKTEIXKKZSEUW0APR22K', 'IeTuFkW6mIrPYYUv69RMcB7LYDzlOkop5hc2df5H',
               base_url='https://paper-api.alpaca.markets', )

    # df=api.polygon.historic_agg_v2("GOOGL",1, tradeapi.rest.TimeFrame.Hour, "2020-02-04", "2021-02-08")

    data = api.get_bars("TSLA", tradeapi.rest.TimeFrame.Day, "2020-03-02", "2021-04-23", adjustment='raw').df
    data.index.rename("Date")

    return (data)
