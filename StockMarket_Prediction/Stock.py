# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:36:52 2022

@author: amaad
"""

import streamlit as st
import datetime
import plotly.express as px
import plotly.graph_objects as go

#Importing Libreries

import pandas as pd

from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings


# df=pd.read_csv('TataDateClose.csv')
#Importing Libreries
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import yfinance as yf
import pandas as pd

st.title('Stock Market Prediction')

tickers = ('INFY', 'WIPRO.NS', 'TATAMOTORS.NS', 'RELIANCE.NS', 'TCS', 'SBIN.NS', 'IBM', 'BHARTIARTL.NS', 'HDB', 'SUNPHARMA', 'DRREDDY')

dropdown = st.multiselect('Pick your assets', tickers)

start = st.date_input('Start', value = pd.to_datetime('2016-01-01'))
end = st.date_input('End', value=pd.to_datetime('today'))



        # Visualizations
def relativeret(df):
    rel = df.pct_change()
    cumret = (1 + rel).cumprod() - 1
    cumret = cumret.fillna(0)
    return cumret
    

if len(dropdown) > 0:
    # df = yf.download(dropdown, start,end)['Close']
    df = relativeret(yf.download(dropdown, start,end)['Close'])
    st.subheader('Close Prices of {}'.format(dropdown))
    st.line_chart(df)

import matplotlib.pyplot as plt 
if len(dropdown) > 0:
    # df = yf.download(dropdown, start,end)['Close']
    st.subheader('200 Days Moving Averages of {}'.format(dropdown))
    fig = plt.figure(figsize=(12,6))
#     ma100 = df.rolling(100).mean()
    ma200 = df.rolling(200).mean()
#     plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df)
    plt.legend()
    st.pyplot(fig)





  
df=pd.read_csv('TataDateClose.csv')

hwmModel=ExponentialSmoothing(df['Close'],seasonal='mul',trend='add',seasonal_periods=24).fit()


def main():
    st.title("Tatamotors Stock Price Predictor")
    st.info("Let us predict the Price of Stock for the Future")
  
    s = datetime.date(2022,1,1)
    e = st.date_input("Enter the ending Date to Predict the Stock Prices")
    diff=( (e-s).days+1)
   
    
   
    if st.button("PREDICT"):
        index_future_dates=pd.date_range(start= s ,end= e)
        pred=hwmModel.forecast(diff).rename('')
        pred.index=index_future_dates
        df = pd.DataFrame(pred)
        
        st.dataframe(df)
        
        st.line_chart(df)
        

if __name__ == '__main__':
    main()   
    
    
    
user_input = st.text_input('Input your choice Stock Ticker To See Basic Statistics: Data from 2016-2021', 'AAPL') #Bydefault keeping it for AAPL stock
import pandas_datareader as data
df = data.DataReader(user_input, 'yahoo', start, end)
#Describing Data
# st.subheader('Data from 2016 to 2021')
st.subheader('{}'.format(user_input))
st.write(df.describe())
        
        
    
