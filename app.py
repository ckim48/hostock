from flask import Flask, render_template, redirect, url_for, flash
import pandas_datareader as pdr
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import nltk
from xmlrpc.client import Boolean
from flask import request
import cv2
import numpy as np
import os
import sys
import json

app = Flask(__name__)
app.secret_key = b'_5#y2L'








def macd_strategy(symbol, start, end):
    """ Backtesting simulation of macd strategy
    Parameters:
    symbol (str): symbol of a stock
    start (datetime): starting date of the backtest
    end (datetime): last date of the backtest

    Returns:
    Dataframe that includes the total amound of asset when using the strategy and drawdown of the strategy. 
    """
    #get data
    price = pdr.get_data_yahoo(symbol, start, end)
    price = price.drop(['Volume', 'Adj Close'], 1)

    #macd calculations
    exp1 = price.Close.ewm(span = 12, adjust=False).mean()
    exp2 = price.Close.ewm(span = 26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span = 9, adjust=False).mean()

    #add column for entries
    price['Long'] = macd > signal
    # profit calculation for MACD
    money = 10000
    exchange = 0
    wins= 0
    asset = [10000]
    numb=0
    drawdowns = [0]
    difference = []
    for i in range(1, len(price)):
        
        if price['Long'][i] == True:
            #buys if macd is above signal and not bought yet
            if numb==0:
                buy = price['Close'][i]
                numb = money//buy
                money-=numb*buy
            # continue if already bought
            else:
                asset.append(money + (price['Close'][i]*numb))
                drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
                continue
        else:
            #sell if macd is bellow signal line and didn't sell the stocks yet
            if numb!=0 : 
                sell = price['Close'][i]
                money += (sell)*numb
                exchange +=1
                if sell>buy:
                    wins +=1
                difference.append(sell-buy)
                numb=0
            else: 
                #continue if already sold
                asset.append(money + (price['Close'][i]*numb))
                drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
                continue
        
        asset.append(money + (price['Close'][i]*numb))
        drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
    price['MACD']=asset
    price['DD']=drawdowns

    STARTING_BALANCE = 10000
    #daily return
    price['Return'] = price.Close / price.Close.shift(1)
    price.Return.iat[0] = 1
    price['Bench_Bal'] = STARTING_BALANCE * price.Return.cumprod()
    #calculate drawdown
    price['Bench_Peak'] = price.Bench_Bal.cummax()
    price['Bench_DD'] = price.Bench_Bal - price.Bench_Peak

    bench_dd = round((price.Bench_DD / price.Bench_Peak).min() * 100, 2)
    #print("win rate:", wins/exchange)
    print("strategy maximum drawdown", min(price['DD']))
    print("benchmarck maximum drawdwon", bench_dd)
    return price
    
def rsi_strategy(symbol, start, end):
    """ Backtesting simulation of rsi strategy
    Parameters:
    symbol (str): symbol of a stock
    start (datetime): starting date of the backtest
    end (datetime): last date of the backtest

    Returns:
    Dataframe that includes the total amound of asset when using the strategy and drawdown of the strategy. 
    """
    #get data
    price = pdr.get_data_yahoo(symbol, start, end)
    price = price.drop(['Volume', 'Adj Close'], 1)
    rsi = price.ta.rsi(close='Close', length=14, append=True, signal_indicators = True, xa=70, xb=30)
    RSIs=[]
    for i in price['RSI_14_B_30']:
        RSIs.append(Boolean(i))
    RSI= pd.Series(RSIs)
    # profit calculation for RSI
    money = 10000
    exchange = 0
    wins= 0
    drawdowns = [0]
    asset = [10000]
    numb=0
    rsi_index = []
    for i in range(1, len(price)):
        if price['RSI_14_B_30'][i]==1 and numb==0:
            rsi_index.append(i)
            buy = price['Close'][i]
            numb = money//buy
            money-=numb*buy
            asset.append(money + (price['Close'][i]*numb))
            drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
            continue

        if price['RSI_14_A_70'][i]==1 and numb !=0:
            sell = price['Close'][i]
            money += (sell)*numb
            exchange +=1
            if sell>buy:
                wins +=1
            numb=0
            asset.append(money + (price['Close'][i]*numb))
            drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
            continue
            
        asset.append(money + (price['Close'][i]*numb))
        drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))

        
    price['RSI_strategy'] = asset
    price['DD']=drawdowns
    print("strategy maximum drawdown", min(price['DD']))
    return price
def bollinger_band_strategy(symbol, start, end):
    """ Backtesting simulation of bollinger strategy
    Parameters:
    symbol (str): symbol of a stock
    start (datetime): starting date of the backtest
    end (datetime): last date of the backtest

    Returns:
    Dataframe that includes the total amound of asset when using the strategy and drawdown of the strategy. 
    """
    def get_sma(prices, rate):
        return prices.rolling(rate).mean()
    price = pdr.DataReader(symbol, 'yahoo', start, end)
    closing_prices = price['Close'] 
    ma = get_sma(closing_prices, 20)
    def get_bollinger_bands(prices, rate=20):
        sma = get_sma(prices, rate)
        std = prices.rolling(rate).std()
        bollinger_up = sma + std * 2 # Calculate top band
        bollinger_down = sma - std * 2 # Calculate bottom band
        return bollinger_up, bollinger_down

    bollinger_up, bollinger_down = get_bollinger_bands(closing_prices)

    # backtrading for bollinger bands
    money = 10000
    exchange = 0
    wins= 0
    asset = [10000]
    numb=0
    bollinger_up, bollinger_down = get_bollinger_bands(price['Close'])

    for i in range(1, len(price)):
        if bollinger_down[i]:
            if price['Close'][i] < bollinger_down[i] and numb==0:
                buy = price['Close'][i]
                numb = money//buy
                money-=numb*buy
            elif price['Close'][i] > bollinger_up[i] and numb!=0:
                sell = price['Close'][i]
                money += (sell)*numb
                exchange +=1
                if sell>buy:
                    wins +=1
                numb=0
            else:
                asset.append(money + (price['Close'][i]*numb))
                continue
        else:
            asset.append(money + (price['Close'][i]*numb))
            continue
        asset.append(money + (price['Close'][i]*numb))

    price['Bollinger'] = asset
    return price

# profit calculation for volatility breakout strategy
def breakout_strategy(symbol, start, end,k):
    """ Backtesting simulation of vollatility breakout strategy
    Parameters:
    symbol (str): symbol of a stock
    start (datetime): starting date of the backtest
    end (datetime): last date of the backtest
    k (float): k is a number between 0 and 1. The strategy will buy when today's price increases by (yesterday's vollatility) * k

    Returns:
    Dataframe that includes the total amound of asset when using the strategy and drawdown of the strategy. 
    """
    price = pdr.get_data_yahoo(symbol, start, end)
    price = price.drop(['Volume', 'Adj Close'], 1)
    money = 10000
    exchange = 0
    wins= 0
    asset = [10000]
    numb=0
    profit=0
    for i in range( len(price)-1):
        volatility = abs(price['High'][i-1]-price['Low'][i-1])
        k=0.3
        if price['High'][i]>price['Open'][i]+(volatility*k):
            buy = price['Open'][i]+(volatility*k)
            numb = money//buy
            sell = price['Close'][i]
            profit = (sell-buy)*numb
            exchange +=1
            if sell>buy:
                wins +=1

        numb=0
        money +=profit
        
        profit=0
        asset.append(money)
    price['breakout'] = asset
    return price
def MACD_BREAKOUT_STRATEGY(symbol, start, end, k):
    """ Backtesting simulation of macd and breakout combined strategy
    Parameters:
    symbol (str): symbol of a stock
    start (datetime): starting date of the backtest
    end (datetime): last date of the backtest
    k (float): k is a number between 0 and 1. The strategy will buy when today's price increases by (yesterday's vollatility) * k

    Returns:
    Dataframe that includes the total amound of asset when using the strategy and drawdown of the strategy. 
    """
    
    #get data
    price = pdr.get_data_yahoo(symbol, start, end)
    price = price.drop(['Volume', 'Adj Close'], 1)

    #macd calculations
    exp1 = price.Close.ewm(span = 12, adjust=False).mean()
    exp2 = price.Close.ewm(span = 26, adjust=False).mean()
    exp3 = price.Close.ewm(span = 26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span = 9, adjust=False).mean()

    #add column for entries
    price['Long'] = macd > signal
    price['200ema']=exp3
    # profit calculation for MACD
    money = 10000
    exchange = 0
    wins= 0
    asset = [10000]
    numb=0
    difference = []
    for i in range(1, len(price)):
        
        if price['Long'][i] == True:
            
            if numb==0:
                buy = price['Close'][i]
                numb = money//buy
                money-=numb*buy
            else:
                asset.append(money + (price['Close'][i]*numb))
                continue
        if price['High'][i]>price['Open'][i]+(price['High'][i-1]-price['Low'][i-1] )*k :
            if numb==0:
                buy = price['Open'][i]+(price['High'][i-1]-price['Low'][i-1] )*k
                numb = money//buy
                money-=numb*buy
            else:
                asset.append(money + (price['Close'][i]*numb))
                continue
        else:
           
            if numb!=0 : 
                sell = price['Close'][i]
                money += (sell)*numb
                exchange +=1
                if sell>buy:
                    wins +=1
                difference.append(sell-buy)
                numb=0
            else: 
                
                asset.append(money + (price['Close'][i]*numb))
                continue
        asset.append(money + (price['Close'][i]*numb))
    price['MACD']=asset
    return price

@app.route('/', methods = ['POST', 'GET'])
def index():
	return render_template('index.html')



if __name__ == '__main__':
	app.run(debug = True)

