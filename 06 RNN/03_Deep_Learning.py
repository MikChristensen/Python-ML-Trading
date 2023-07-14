# %% [markdown]
# # Case Study A-Z: A Machine Learning powered Strategy (DNN)

# %% [markdown]
# _Disclaimer: <br>
# The following illustrative example is for general information and educational purposes only. <br>
# It is neither investment advice nor a recommendation to trade, invest or take whatsoever actions.<br>
# The below code should only be used in combination with an Oanda/FXCM Practice/Demo Account and NOT with a Live Trading Account._

# %% [markdown]
# ## Getting and Preparing the Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Preprocessing data
from talib import abstract
from sklearn.preprocessing import RobustScaler, MinMaxScaler 
import math # Mathematical functions 


# %%
data = pd.read_csv("DNN_data.csv", parse_dates = ["time"], index_col = "time")

# %%
data 

# %%
data.info()

# %%
symbol = data.columns[0]
symbol

# %%
data.plot(figsize = (12, 8))
plt.show()

# %%
#data["returns"] = data[symbol] / data[symbol].shift()
data["returns"] = np.log(data[symbol] / data[symbol].shift())

# %%
data

# %% [markdown]
# ## Adding Label/Features

# %%
window = 50

# %%
df = data.copy()
df["dir"] = np.where(df["returns"] > 0, 1, 0)
df["sma"] = df[symbol].rolling(window).mean() - df[symbol].rolling(150).mean()
df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()) / df[symbol].rolling(window).std()
df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
df["mom"] = df["returns"].rolling(3).mean()
df["vol"] = df["returns"].rolling(window).std()
df.dropna(inplace = True)

# %%
df

# %% [markdown]
# ## Adding Features

# %%
lags = 5

# %%
cols = []
features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

# %%

for lag in range(1, lags + 1):
    col = "lag{}".format(lag)
    df[col] = df["returns"].shift(lag)
    cols.append(col)
          
df['Ask_Dir'] = np.where(df['returns'].shift(-1) > df.returns, 1, 0)
cols.append('Ask_Dir')

df['SMA'] = abstract.SMA(df['returns'], timeperiod=12)
cols.append('SMA')
df['SMA_Dir'] = np.where(df['SMA'].shift(-1) > df.SMA, 1, 0)
cols.append('SMA_Dir')

df['RSI'] = abstract.RSI(df['returns'], timeperiod=12)
cols.append('RSI')
df['RSI_Dir'] = np.where(df['RSI'].shift(-1) > df.RSI, 1, 0)
cols.append('RSI_Dir')

df['fastk'], df['fastd'] = abstract.STOCHRSI(df["returns"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
cols.append('fastk')
cols.append('fastd')
df['fastk_Dir'] = np.where(df['fastk'].shift(-1) > df.fastk, 1, 0)
cols.append('fastk_Dir')
df['fastd_Dir'] = np.where(df['fastd'].shift(-1) > df.fastd, 1, 0)    
cols.append('fastd_Dir')            
   
            
df.dropna(inplace = True)

# %%
df.columns

# %%
df.info()

# %%
len(cols)

# %%


# %% [markdown]
# ## Splitting into Train and Test Set

# %%
len(df)

# %%
split = int(len(df)*0.66)
split

# %%
train = df.iloc[:split].copy()
train

# %%
test = df.iloc[split:].copy()
test

# %%
train

# %% [markdown]
# ## Feature Scaling (Standardization)

# %%
train[cols]

# %%
mu, std = train.mean(), train.std() # train set parameters (mu, std) for standardization

# %%
std

# %%
train_s = (train - mu) / std # standardization of train set features

# %%
train_s.describe()

# %%
df.describe

# %% [markdown]
# ## Creating and Fitting the DNN Model

# %%
from DNNModel import *

# %%
# fitting a DNN model with 3 Hidden Layers (50 nodes each) and dropout regularization

set_seeds(100)
model = create_model(hl = 3, hu = 50, dropout = True, input_dim = len(cols))
model.fit(x = train_s[cols], y = train["dir"], epochs = 50, verbose = False,
          validation_split = 0.2, shuffle = False, class_weight = cw(train))

# %%
model.evaluate(train_s[cols], train["dir"]) # evaluate the fit on the train set

# %%
pred = model.predict(train_s[cols]) # prediction (probabilities)
pred

# %%
plt.hist(pred, bins = 50)
plt.show()

# %%


# %% [markdown]
# ## Out-Sample Prediction and Forward Testing

# %%
test

# %%
test_s = (test - mu) / std # standardization of test set features (with train set parameters!!!)

# %%
model.evaluate(test_s[cols], test["dir"])

# %%
pred = model.predict(test_s[cols])
pred

# %%
plt.hist(pred, bins = 50);

# %%
test["proba"] = model.predict(test_s[cols])

# %%
test["position"] = np.where(test.proba < 0.47, -1, np.nan) # 1. short where proba < 0.47

# %%
test["position"] = np.where(test.proba > 0.53, 1, test.position) # 2. long where proba > 0.53

# %%
test["position"].min()


# %%
test["position"]

# %%
test.index = test.index.tz_localize("UTC")
test["NYTime"] = test.index.tz_convert("America/New_York")
test["hour"] = test.NYTime.dt.hour

# %%
test["position"] = np.where(~test.hour.between(2, 12), 0, test.position) # 3. neutral in non-busy hours

# %%
test["position"] = test.position.ffill().fillna(0) # 4. in all other cases: hold position

# %%
test.position.value_counts(dropna = False)

# %%
test["strategy"] = test["position"] * test["returns"]

# %%
test["creturns"] = test["returns"].cumsum().apply(np.exp)
test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)

# %%
test["creturns"].describe()

# %%
test[["creturns", "cstrategy"]].plot(figsize = (12, 8))
plt.show()

# %%
ptc = 0.000059

# %%
test["trades"] = test.position.diff().abs()

# %%
test.trades.value_counts()

# %%
test["strategy_net"] = test.strategy - test.trades * ptc

# %%
test["cstrategy_net"] = test["strategy_net"].cumsum().apply(np.exp)

# %%
test[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12, 8))
plt.show()

# %%


# %% [markdown]
# ## Saving Model and Parameters

# %%
model

# %%
mu

# %%
std

# %%
model.save("DNN_model")

# %%
import pickle

# %%
params = {"mu":mu, "std":std}

# %%
params

# %%
pickle.dump(params, open("params.pkl", "wb"))

# %%


# %% [markdown]
# ## Implementation

# %%
import pandas as pd
import numpy as np
import tpqoa
import fxcmpy
from datetime import datetime, timedelta
import time

# %% [markdown]
# __Loading Model and Parameters__

# %%
# Loading the model
import keras
model = keras.models.load_model("DNN_model")

# %%
model

# %%
# Loading mu and std
import pickle
params = pickle.load(open("params.pkl", "rb"))
mu = params["mu"]
std = params["std"]

# %%
mu

# %%
std

# %% [markdown]
# __[FXCM] Implementation__

# %%


# %%
api = fxcmpy.fxcmpy(config_file= "FXCM.cfg")

# %%
col = ["tradeId", "amountK", "currency", "grossPL", "isBuy"]

# %%
class DNNTrader():
    
    def __init__(self, instrument, bar_length, window, lags, model, mu, std, units):
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length) 
        self.tick_data = None
        self.raw_data = None
        self.data = None 
        self.ticks = 0
        self.last_bar = None  
        self.units = units
        self.position = 0
        
        #*****************add strategy-specific attributes here******************
        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        #************************************************************************        
    
    def get_most_recent(self, period = "m1", number = 10000):
        while True:  
            time.sleep(5)
            df = api.get_candles(self.instrument, number = number, period = period, columns = ["bidclose", "askclose"])
            df[self.instrument] = (df.bidclose + df.askclose) / 2
            df = df[self.instrument].to_frame()
            df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()) - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.utcnow()) # NEW -> Start Time of Trading Session
                break
    
    def get_tick_data(self, data, dataframe):
        
        self.ticks += 1
        print(self.ticks, end = " ", flush = True)
        
        recent_tick = pd.to_datetime(data["Updated"], unit = "ms")
        
        if recent_tick - self.last_bar > self.bar_length:
            self.tick_data = dataframe.loc[self.last_bar:, ["Bid", "Ask"]]
            self.tick_data[self.instrument] = (self.tick_data.Ask + self.tick_data.Bid)/2
            self.tick_data = self.tick_data[self.instrument].to_frame()
            self.resample_and_join()
            self.define_strategy() 
            self.execute_trades()
            
    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, 
                                                             label="right").last().ffill().iloc[:-1])
        self.last_bar = self.raw_data.index[-1]  
        
    def define_strategy(self): # "strategy-specific"
        df = self.raw_data.copy()
        
        #******************** define your strategy here ************************
        df = df.append(self.tick_data.iloc[-1]) # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace = True)
        
        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for lag in range(1, lags + 1):
            col = "lag{}".format(lag)
            df[col] = df["returns"].shift(lag)
            cols.append(col)
                
                
        df['Ask_Dir'] = np.where(df['returns'].shift(-1) > df.returns, 1, 0)
        cols.append('Ask_Dir')

        df['SMA'] = abstract.SMA(df['returns'], timeperiod=12)
        cols.append('SMA')
        df['SMA_Dir'] = np.where(df['SMA'].shift(-1) > df.SMA, 1, 0)
        cols.append('SMA_Dir')

        df['RSI'] = abstract.RSI(df['returns'], timeperiod=12)
        cols.append('RSI')
        df['RSI_Dir'] = np.where(df['RSI'].shift(-1) > df.RSI, 1, 0)
        cols.append('RSI_Dir')

        df['fastk'], df['fastd'] = abstract.STOCHRSI(df["returns"], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        cols.append('fastk')
        cols.append('fastd')
        df['fastk_Dir'] = np.where(df['fastk'].shift(-1) > df.fastk, 1, 0)
        cols.append('fastk_Dir')
        df['fastd_Dir'] = np.where(df['fastd'].shift(-1) > df.fastd, 1, 0)    
        cols.append('fastd_Dir')                
                
        df.dropna(inplace = True)
        
        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        df["proba"] = self.model.predict(df_s[self.cols])
        
        print(df.columns )
        
        #determine positions
        df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
        df["position"] = np.where(df.proba < 0.47, -1, np.nan)
        df["position"] = np.where(df.proba > 0.53, 1, df.position)
        df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
        #***********************************************************************
        
        self.data = df.copy()
    
    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING LONG")  
            elif self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING LONG")  
            self.position = 1
        elif self.data["position"].iloc[-1] == -1: 
            if self.position == 0:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING SHORT")  
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units * 2)
                self.report_trade(order, "GOING SHORT")  
            self.position = -1
        elif self.data["position"].iloc[-1] == 0: 
            if self.position == -1:
                order = api.create_market_buy_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL") 
            elif self.position == 1:
                order = api.create_market_sell_order(self.instrument, self.units)
                self.report_trade(order, "GOING NEUTRAL")  
            self.position = 0

    def report_trade(self, order, going):
        time = order.get_time()
        units = api.get_open_positions().amountK.iloc[-1]
        price = api.get_open_positions().open.iloc[-1]
        unreal_pl = api.get_open_positions().grossPL.sum()
        print("\n" + 100* "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | Unreal. P&L = {}".format(time, units, price, unreal_pl))
        print(100 * "-" + "\n")

# %%
trader = DNNTrader("EUR/USD", bar_length = "20min", 
                   window = 50, lags = 5, model = model, mu = mu, std = std, units = 100)

# %%

trader.get_most_recent()
api.subscribe_market_data(trader.instrument, (trader.get_tick_data, ))

# %%
'''
api.unsubscribe_market_data(trader.instrument)
if len(api.get_open_positions()) != 0: # if we have final open position(s) (netting and hedging)
    api.close_all_for_symbol(trader.instrument)
    print(2*"\n" + "{} | GOING NEUTRAL".format(str(datetime.utcnow())) + "\n")
    time.sleep(20)
    print(api.get_closed_positions_summary()[col])
    trader.position = 0
'''

# %%
#trader.data

# %%
#api.close()

# %%



