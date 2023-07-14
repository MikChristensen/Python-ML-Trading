# %%
# https://github.com/zbanga/FXCM_currency/blob/master/Automated%20Algo.ipynb

# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import fxcmpy
import time
from talib import abstract
import datetime as dt
#import datetime
import cufflinks as cf  # Cufflinks
cf.set_config_file(offline=True)  # set the plotting mode to offline

import plotly.graph_objs as go

# %% [markdown]
# ## Download data for model generation

# %%
api = fxcmpy.fxcmpy(config_file='FXCM.cfg')

# %%
# More documentation http://fxcmpy.tpq.io/
# Calling API to get data 
# change start and stop times
# Instrument must be one of ('EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', '
# EUR/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY',
#'AUD/JPY', 'USD/CNH', 'FRA40', 'GER30', 'UK100', 'US30', 'USDOLLAR', 'XAU/USD', 'XAG/USD').

df_candles = api.get_candles('EUR/USD', period='m5',
                         start=dt.datetime(2018, 2, 7),
                          stop=dt.datetime(2018, 2, 22))

# %%
#Putting data into a dataframe with a midclose 
df = pd.DataFrame(df_candles[['askclose','bidclose']].mean(axis=1),columns=['Mid'])

# %% [markdown]
# ## Generate features

# %%
def generate_features(df):
    df["Returns"] = np.log(df['Mid'] / df['Mid'].shift(1))
    lags = 5
    cols = []
    for lag in range(1, lags + 1):
        col = 'lag_%s' % lag
        df[col] = df['Returns'].shift(lag)
        cols.append(col)
    
    df['Ask_Dir'] = np.where(df['Returns'].shift(-1) > df['Returns'], 1, 0)
    cols.append('Ask_Dir')

    df['SMA'] = abstract.SMA(df['Returns'], timeperiod=12)
    cols.append('SMA')
    df['SMA_Dir'] = np.where(df['SMA'].shift(-1) > df.SMA, 1, 0)
    cols.append('SMA_Dir')

    df['RSI'] = abstract.RSI(df['Returns'], timeperiod=12)
    cols.append('RSI')
    df['RSI_Dir'] = np.where(df['RSI'].shift(-1) > df.RSI, 1, 0)
    cols.append('RSI_Dir')

    df['fastk'], df['fastd'] = abstract.STOCHRSI(df['Returns'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    cols.append('fastk')
    cols.append('fastd')
    df['fastk_Dir'] = np.where(df['fastk'].shift(-1) > df.fastk, 1, 0)
    cols.append('fastk_Dir')
    df['fastd_Dir'] = np.where(df['fastd'].shift(-1) > df.fastd, 1, 0)    
    cols.append('fastd_Dir')    

    
    df.dropna(inplace=True)
    return df , cols

# %%

print(len(df))

#df , cols = generate_features(df)

# %% [markdown]
# ## Split data

# %%
train_x, test_x, train_y, test_y = train_test_split(
    df[cols],                   
    np.sign(df['Returns']),
    shuffle=True,
    test_size=0.50, random_state=1111)

train_x.sort_index(inplace=True)
train_y.sort_index(inplace=True)
test_x.sort_index(inplace=True)
test_y.sort_index(inplace=True)

# %%
train_y

# %% [markdown]
# ## Build model

# %%
from sklearn.neural_network import MLPClassifier

# %%
model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
model.fit(train_x, train_y)

train_pred = model.predict(train_x)

# %% [markdown]
# ## Validate model

# %%
print('accuracy_score for testing')
accuracy_score(train_y, train_pred)

# %%
test_pred = model.predict(test_x)

# %%
print('accuracy_score for testing')
print(accuracy_score(test_y, test_pred))

# %% [markdown]
# # AUTOMATED TRADING

# %%
candles = api.get_candles('EUR/USD', period = "m5", number = 1000)
data = pd.DataFrame(candles[['askclose','bidclose']].mean(axis=1),columns=['Mid'])
data , cols = generate_features(data)

# %%
model.predict(data[cols])[:10]

# %%
to_show = ['tradeId','amountK','currency','grossPL','isBuy']
#Initalising Variables
ticks = 0 
position = 0 
tick_data = pd.DataFrame()
tick_resam = pd.DataFrame()
#Changable Variables 
unit_size= 2.5

# %%
def automated_trading(data,df):
    global lags, model, ticks
    global tick_data , tick_resam , to_show
    global position
    ticks +=1
    t = dt.datetime.now()
    if ticks == 1:
        print('starting now at %s' %(str(t.time()) ))
    if ticks % 1000 == 0:
        print('%3d | %s | %7.5f | %7.5f' % (ticks, str(t.time()) , data['Rates'][0],data['Rates'][1]))
    
    now = dt.now()

    current_time = now.strftime("%H:%M:%S")
    print("Live signal at =", current_time)
    
    
    
    #COLLECTING TICK DATA 
    tick_data = tick_data.append(pd.DataFrame(
        {'Bid':data['Rates'][0],'Ask': data['Rates'][1],
        'High': data['Rates'][2],'Low': data['Rates'][3]},index=[t]))
    
    
    #Resample Tick Data 
    tick_resam = tick_data[['Bid', 'Ask']].resample('5Min', label='right').last().ffill()
    tick_resam['Mid'] = tick_resam.mean(axis=1)
    
    if len(tick_resam) > lags + 2:
        #Generating Signal 
    #print("Starting Signal Generation")
    
        tick_resam, cols = generate_features(tick_resam, lags)
        tick_resam['Prediction'] = model.predict(tick_resam[cols])
        #Generating a long position
        
        if tick_resam['Prediction'].iloc[-2] >= 0 and position == 0:
            
            print('Going Long for the first time')
            position = 1 
            order = api.create_market_buy_order('EUR/USD', unit_size)
            #trade = True
            #time.sleep(60)
            
        elif tick_resam['Prediction'].iloc[-2] >= 0 and position == -1:
            #api.close_all_for_symbol('EUR/USD')
            print('Going Long ')
            position = 1 
            
            tradeId = api.get_open_trade_ids()[0]
            pos = api.get_open_position(tradeId)
            pos.close()
            order = api.create_market_buy_order('EUR/USD', unit_size)
            #trade = True
            #time.sleep(60)
        #Entering a short position
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position == 0:
            print('Going Short for the first time ')
            position = -1 
            order = api.create_market_sell_order('EUR/USD', unit_size)
            #trade = True
            #time.sleep(60)
        elif tick_resam['Prediction'].iloc[-2] <= 0 and position == 1:
            print('Going Short ')
            #api.close_all_for_symbol('EUR/USD')
            
            tradeId = api.get_open_trade_ids()[0]
            pos = api.get_open_position(tradeId)
            pos.close()
            position = -1 
            order = api.create_market_sell_order('EUR/USD', unit_size)

# %%


# %%
print('start')
api.subscribe_market_data('EUR/USD',(automated_trading,))

# %%



