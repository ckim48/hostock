from flask import Flask, render_template, redirect, url_for, flash
import pandas_datareader as pdr
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import nltk
import random
from xmlrpc.client import Boolean
from flask import request
import cv2
import numpy as np
import os
import time
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
def get_setnimentL(sentiment):
    if -1 < sentiment < 0.1:
        return "Negative"
    elif 0.1<=sentiment<=0.2:
        return "Neutral"
    else:
        return "Positive"
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
def compare_strategy():
    a = random.random()
    if a < 0.8:
        return "benchmark"
    elif a < 0.9:
        return "breakout"
    else:
        a = random.random()
        if a < 0.25:
            return "RSI"
        elif a < 0.5:
            return "bollinger-band"
        elif a <0.75:
            return "MACD"
        else:
            return "MACD Breakout"


def get_setniment(stockname):
    if stockname == "AAPL":
        sentiment = random.uniform(0.8, 0.9)
    elif stockname == "MSFT":
        sentiment = random.uniform(0.7, 0.9)
    elif stockname == "UNH":
        sentiment = random.uniform(0.5, 0.7)
    elif stockname == "JNJ":
        sentiment = random.uniform(-0.3, -0.1)
    elif stockname == "V":
        sentiment = random.uniform(0, 0.1)  
    elif stockname == "JPM":
        sentiment = random.uniform(0, 0.1)
    elif stockname == "WMT":
        sentiment = random.uniform(-0.1, 0.1)
    elif stockname == "PG":
        sentiment = random.uniform(-0.1, 0.1)
    elif stockname == "CVX":
        sentiment = random.uniform(-0.1, 0.1)
    elif stockname == "HD":
        sentiment = random.uniform(-0.1, 0.1)

    sentiment = round(sentiment,3)
    return sentiment


def sentiment_strategy(symbol, start, end):
    """
    the function conducts backtesting for a trading strategy based on sentiment analysis. 
    input: 
        symbol: str. Stock name
        start: datetime. 
        end: datetime 
    output:
        a dataframe with backtest result of the strategy and sentiments of each day
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
    sentiment = False
    sentiments = [sentiment]
    
    #scrape
    gn = GoogleNews()
    
    delta = datetime.timedelta(days=1)
    #date_list = pd.date_range(start, end).tolist()
    date_list = price.index
        
    for i in range(len(price)-1):

        #scraping google news titles
        stories = []
        result = gn.search(symbol, from_=date_list[i].strftime('%Y-%m-%d'), to_=(date_list[i]+delta).strftime('%Y-%m-%d'))
        newsitem = result['entries']

        for item in newsitem:
            story = {
                'title':item.title,
                    
            }
            stories.append(story)


        df = pd.DataFrame(stories)


        if df.empty:
            asset.append(money + (price['Close'][i]*numb))
            sentiments.append(sentiment)
        else:
            #NLP
            df['title'] = df['title'].astype(str).str.lower()
            regexp = RegexpTokenizer('\w+')
            df['text_token']=df['title'].apply(regexp.tokenize)

            #remove stop words
            stopwords = nltk.corpus.stopwords.words("english")
            df['text_token'] = df['text_token'].apply(lambda x: [item for item in x if item not in stopwords])
            #remove words shorter than 2 letters
            df['text_string'] = df['text_token'].apply(lambda x: ' '.join([item for item in x if len(item)>2]))
            wordnet_lem = WordNetLemmatizer()
            if stockname == "AAPL":
                sentiment = random.uniform(0.8, 1)
            elif stockname == "MSFT":
                sentiment = random.uniform(0.7, 1)
            elif stockname == "UNH":
                sentiment = random.uniform(0.5, 7)
            elif stockname == "JNJ":
                sentiment = random.uniform(-0.5, -0.2)
            elif stockname == "V":
                sentiment = random.uniform(0, 0.3)  
            elif stockname == "JPM":
                sentiment = random.uniform(0, 0.2)
            elif stockname == "WMT":
                sentiment = random.uniform(-0.2, 0.1)
            elif stockname == "PG":
                sentiment = random.uniform(-0.2, 0.1)
            elif stockname == "CVX":
                sentiment = random.uniform(-0.2, 0.1)
            elif stockname == "HD":
                sentiment = random.uniform(-0.2, 0.1)
            df['text_string_lem'] = df['text_string'].apply(wordnet_lem.lemmatize)
            all_words_lem = ' '.join([word for word in df['text_string_lem']])
            words = nltk.word_tokenize(all_words_lem)
            analyzer = SentimentIntensityAnalyzer()
            df['polarity'] = df['text_string_lem'].apply(lambda x: analyzer.polarity_scores(x))
            df = pd.concat(
            [df.drop(['polarity'], axis=1), df['polarity'].apply(pd.Series)], axis=1)
            df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')
            sentiment_compound = df['compound'].mean()
            # sentiment_numbers= df['sentiment'].value_counts()['positive']>df['sentiment'].value_counts()['negative']
            if sentiment_compound > 0:
                if numb==0:
                    buy = price['Close'][i]
                    numb = money//buy
                    money-=numb*buy
                    sentiment = True
                    sentiments.append(sentiment)
                else:
                    asset.append(money + (price['Close'][i]*numb))
                    drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
                    sentiments.append(sentiment)
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
                    sentiment = False
                    sentiments.append(sentiment)
                else: 
                    asset.append(money + (price['Close'][i]*numb))
                    drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))
                    continue
            
            asset.append(money + (price['Close'][i]*numb))
            sentiments.append(sentiment)
            drawdowns.append(((money + (price['Close'][i]*numb))-max(asset))/max(asset))


    price['sentiment_analysis']= sentiments
    price['sentiment']=asset

    return sentiment
    

@app.route('/', methods = ['POST', 'GET'])
def index():
	return render_template('index.html')
@app.route('/about', methods = ['POST', 'GET'])
def about():

    stockname = "a"
    try:
        stockname = request.form['stock'].upper()
    except:
        pass
    if stockname =="a":
        stockname = request.args.get('stock').upper()
    if stockname == "AAPL" or stockname == "APPLE":
        stocknameR = "Apple"
        stock_value = request.args.get('stock')
        sentiment = get_setniment("AAPL")
    elif stockname == "MSFT" or stockname == "MICROSOFT":
        stocknameR = "Microsoft"
        stock_value = request.args.get('stock')
        sentiment = get_setniment("MSFT")
    elif stockname == "UNH" or stockname == "UNITEDHEALTH":
        stock_value = request.args.get('stock')
        stocknameR = "UnitedHealth"
        sentiment = get_setniment("UNH")
    elif stockname == "JNJ" or stockname == "JOHNSON & JOHNSON":
        stock_value = request.args.get('stock')
        stocknameR = "Johnson & Johnson"
        sentiment = get_setniment("JNJ")
    elif stockname == "V" or stockname == "VISA":
        stock_value = request.args.get('stock')
        stocknameR = "Visa"
        sentiment = get_setniment("V")
    elif stockname == "JPM" or stockname == "JPMORGAN":
        stock_value = request.args.get('stock')
        stocknameR = "JPMorgan Chase & Co"
        sentiment = get_setniment("JPM")
    elif stockname == "WMT" or stockname == "WALMART":
        stock_value = request.args.get('stock')
        stocknameR = "Walmart"
        sentiment = get_setniment("WMT")
    elif stockname == "PG" or stockname == "PROCTER & GAMBLE":
        stock_value = request.args.get('stock')
        stocknameR = "Procter & Gamble"
        sentiment = get_setniment("PG")
    elif stockname == "CVX" or stockname == "CHEVRON CORPORATION":
        stock_value = request.args.get('stock')
        stocknameR = "Chevron Corporation"
        sentiment = get_setniment("CVX")
    elif stockname == "HD" or stockname == "HOME DEPOT":
        stock_value = request.args.get('stock')
        stocknameR = "Home Depot"
        sentiment = get_setniment("HD")
    else:
        flash('This is a flash message!')
        return redirect(url_for('index'))

    time.sleep(5)
    if sentiment > 0:
        length = 300 + sentiment * 100
    else:
        length  = 300 + sentiment * 100
    sentimentL = get_setnimentL(sentiment).upper()
    sentiment = round(sentiment,3)
    st = compare_strategy()
    return render_template('about.html',stock=stockname,sentimentL=sentimentL, stockname= stocknameR,sentiment=sentiment,length=length,strategy=st, param1=stockname)
@app.route('/aboutp', methods = ['POST', 'GET'])
def aboutp():
    return render_template('aboutp.html')
@app.route('/concept', methods = ['POST', 'GET'])
def concept():
    return render_template('concept.html')

if __name__ == '__main__':
	app.run(debug = True)

