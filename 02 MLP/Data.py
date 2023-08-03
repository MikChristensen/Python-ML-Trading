import pandas as pd
import numpy as np
import yfinance as yf

class get_data():
    def get_yfinance_hist(self, symbol = "EURUSD=X", period ='1mo', interval ='15m' ):
        df_raw = yf.download(symbol,period= period ,interval=interval)
        df_raw["returns"] = np.log(df_raw.Close.div(df_raw.Close.shift(1)))
        df_raw.dropna(how="any",inplace=True)
        df_raw["target"] = np.where(df_raw["returns"].shift(-1) > 0, 1, 0)
        
        return df_raw        