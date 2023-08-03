import pandas as pd
import numpy as np
import talib
from talib import abstract
from sklearn.preprocessing import StandardScaler

class Preprocessing():

    #standardizing the data
    def get_data_scaled(self, df, cols):
        ss = StandardScaler() #standardizing the data
        X = ss.fit_transform(df[cols])
    
        return X
    
    def get_feature(self, DF):
        df = DF.copy()

        lags = 200
        nSMAs = 200
        nRSIs = 20
        nADXs = 20

        cols = []

        for lag in range(1, lags + 1):
            col = "lag{}".format(lag)
            df[col] = df["returns"].shift(lag)
            cols.append(col)

        # MACD signals
        df["ma_fast"] = df["Adj Close"].ewm(span=14, min_periods=14).mean()
        cols.append("ma_fast")
        df["ma_slow"] = df["Adj Close"].ewm(span=17, min_periods=17).mean()
        cols.append("ma_slow")
        df["macd"] = df["ma_fast"] - df["ma_slow"]
        cols.append("macd")
        df["signal"] = df["macd"].ewm(span=2, min_periods=17).mean()
        cols.append("signal")

        df['MACD_Signal'] = np.where(df.macd > df.signal, 1, -1)
        cols.append('MACD_Signal')

        # SMA signals
        for nSMA in range(5, nSMAs, 20):
            col1 = "SMA{}".format(nSMA)
            df[col1] = abstract.SMA(df['Close'], timeperiod=nSMA)
            cols.append(col1)
            col2 = "SMA_Signal{}".format(nSMA)
            df[col2] = np.where(df.Close > df[col1], 1, -1)
            cols.append(col2)

        # SMA Crossover signals
        df['fast'] = abstract.SMA(df['returns'], timeperiod=5)
        cols.append('fast')
        df['slow'] = abstract.SMA(df['returns'], timeperiod=8)
        cols.append('slow')
        df['CrossOver_Signal'] = np.where(df.fast > df.slow, 1, -1)
        cols.append('CrossOver_Signal')



        df = df.dropna()

        return df, cols


