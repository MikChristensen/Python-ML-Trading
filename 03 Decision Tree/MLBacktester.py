
import pandas as pd
import numpy as np
from talib import abstract
import yfinance as yf
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.style.use("seaborn")

class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self, symbol, interval, period, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.tc = tc
        self.model = DecisionTreeClassifier()
        self.results = None
        self.getHistoryYfinance(symbol, interval, period)
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc)
    '''                         
    def get_data(self):
        #Imports the data from five_minute_pairs.csv (source can be changed).
        
        raw = pd.read_csv("five_minute_pairs.csv", parse_dates = ["time"], index_col = "time")
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
        
                             
    def get_data(self):
         #Imports the data from five_minute_pairs.csv (source can be changed).
                
        raw = pd.read_csv("EURUSD=X_Daily.csv", parse_dates = ["Date"], index_col = "Date")
        raw.rename(columns={"Close": "price"}, inplace=True)
        raw["returns"] = np.log(raw['price'] / raw['price'].shift(1))
        raw = raw.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        raw.dropna(inplace=True)
        self.data = raw
    '''    
    def getHistoryYfinance(self, symbol, interval, period):

        data = yf.download(  # or pdr.get_data_yahoo(...
        # tickers list or string as well
        tickers = symbol,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        # period = "ytd",
        period = period,

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = interval,

        # group by ticker (to access via data['SPY'])
        # (optional, default is 'column')
        group_by = 'ticker',

        # adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = False,

        # download pre/post regular market hours data
        # (optional, default is False)
        prepost = False,

        # use threads for mass downloading? (True/False/Integer)
        # (optional, default is True)
        threads = True,

        # proxy URL scheme use use when downloading?
        # (optional, default is None)
        proxy = None
        )
        
        data.rename(columns={"Close": "price"}, inplace=True)
        data["returns"] = np.log(data['price'] / data['price'].shift(1))
        data = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
        data.dropna(inplace=True)
        self.data = data
        self.start = min(data.index) 
        self.end = max(data.index) 
        
        return data        
        
        
        
                                    
    def split_data(self, start, end):
        ''' Splits the data into training set & test set.
        '''
        data = self.data.loc[start:end].copy()
        return data
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for training set and test set.
        '''
        self.data_subset = self.split_data(start, end)
        self.feature_columns = []
        
        self.data_subset['Y'] = self.data_subset['returns'].pct_change(periods=10)
        for i,row in self.data_subset.iterrows():
            if row['Y'] >= 0.001:# and row['target'] == 1 :
                price = 1
                self.data_subset._set_value(i,'Y',price)
            elif row['Y'] <= -0.001:# and row['target'] == 0:
                price = -1
                self.data_subset._set_value(i,'Y',price)
            else:
                price = 0
                self.data_subset._set_value(i,'Y',price)        
        
        
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
            
        self.data_subset['Ask_Dir'] = np.where(self.data_subset['price'].shift(-1) > self.data_subset.price, 1, 0)
        self.feature_columns.append('Ask_Dir')

        self.data_subset['SMA'] = abstract.SMA(self.data_subset['price'], timeperiod=12)
        self.feature_columns.append('SMA')
        self.data_subset['SMA_Dir'] = np.where(self.data_subset['SMA'].shift(-1) > self.data_subset.SMA, 1, 0)
        self.feature_columns.append('SMA_Dir')

        self.data_subset['RSI'] = abstract.RSI(self.data_subset['price'], timeperiod=12)
        self.feature_columns.append('RSI')
        self.data_subset['RSI_Dir'] = np.where(self.data_subset['RSI'].shift(-1) > self.data_subset.RSI, 1, 0)
        self.feature_columns.append('RSI_Dir')
        
        self.data_subset['fastk'], self.data_subset['fastd'] = abstract.STOCHRSI(self.data_subset["price"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        self.feature_columns.append('fastk')
        self.feature_columns.append('fastd')
        self.data_subset['fastk_Dir'] = np.where(self.data_subset['fastk'].shift(-1) > self.data_subset.fastk, 1, 0)
        self.feature_columns.append('fastk_Dir')
        self.data_subset['fastd_Dir'] = np.where(self.data_subset['fastd'].shift(-1) > self.data_subset.fastd, 1, 0)    
        self.feature_columns.append('fastd_Dir')
            
        self.data_subset.dropna(inplace=True)
        
    def fit_model(self, start, end):
        ''' Fitting the ML Model.
        '''
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.feature_columns], self.data_subset["Y"])
        
    def test_strategy(self, train_ratio = 0.7, lags = 5):
        ''' 
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        train_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
        lags: int
            number of lags serving as model features.
        '''
        self.lags = lags
                  
        # determining datetime for start, end and split (for training an testing period)
        full_data = self.data.copy()
        split_index = int(len(full_data) * train_ratio)
        split_date = full_data.index[split_index-1]
        train_start = full_data.index[0]
        test_end = full_data.index[-1]
        
        # fit the model on the training set
        self.fit_model(train_start, split_date)
        
        # prepare the test set
        self.prepare_features(split_date, test_end)
                  
        # make predictions on the test set
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = predict
        
        # calculate Strategy Returns
        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
        # determine the number of trades in each bar
        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
        # calculate cumulative returns for strategy & buy and hold
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - self.results["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
        
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "Logistic Regression: {} | TC = {}".format(self.symbol, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
            
