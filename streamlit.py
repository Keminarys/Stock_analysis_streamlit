import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import streamlit as st


st.set_page_config(layout="wide")

st.title('Stock analyzer')  

pages = ['Prices','Candlesticks','Volume','General Info']

with st.sidebar.expander("General Input"):
  st.write("This section is used for stock analysis")
  ticker = st.text_input('Please enter a ticker')
  period_start = st.date_input('Please enter starting date')
  period_end = st.date_input('Please enter ending date')
  ma_period = st.text_input('Please enter a moving average period')
  interval = st.selectbox('Please choose an interval', ['1d', '1wk', '1mo'])
  plots = st.radio('Select a plot to show', pages)

with st.expander('Scope reminder'):
  st.write(f'Analysis is for {ticker} prices from {period_start} to {period_end} with an interval of {interval} and moving average is based on {ma_period} days.')

period1 = int(time.mktime(period_start.timetuple()))
period2 = int(time.mktime(period_end.timetuple()))

url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(url)

ma_period_int = int(ma_period)
df['SMA'] = df['Close'].rolling(ma_period_int).mean()
df['EMA'] = df['Close'].ewm(span=ma_period_int).mean()

### Volumes
fig3 = px.histogram(df, x="Date", y='Volume', nbins=len(df.Volume), title = ticker+" Volume", width=1500, height=700)
fig3.update_layout(bargap=0.2)

### Candlesticks
fig2 = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig2.update_layout(
    title=ticker+" Candlestick",
    autosize=False,
    width=1500,
    height=700)

### Prices 
fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                    mode='lines',
                    name='Close'))
fig4.add_trace(go.Scatter(x=df["Date"], y=df["SMA"],
                    mode='lines',
                    name='SMA'))
fig4.add_trace(go.Scatter(x=df["Date"], y=df["EMA"],
                    mode='lines', name='EMA'))

fig4.update_layout(
    title=ticker+" Closing/SMA/EMA on "+str(ma_period)+" days",
    autosize=False,
    width=1500,
    height=700)

with st.container():
  if plots == 'Prices' : st.plotly_chart(fig4)
  if plots == 'Candlesticks' : st.plotly_chart(fig2)
  if plots == 'Volume' : st.plotly_chart(fig3)  
