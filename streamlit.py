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

pages = ['Prices','Candlesticks','Volume','Forecast']

st.sidebar.header('Parameters')
ticker = st.sidebar.text_input('Please enter a ticker')
period_start = st.sidebar.date_input('Please enter starting date')
period_end = st.sidebar.date_input('Please enter ending date')
ma_period = st.sidebar.text_input('Please enter a moving average period')
interval = st.sidebar.selectbox('Please choose an interval', ['1d', '1w', '1m'])
plots = st.sidebar.radio('Select a plot to show', pages)

with st.expander('Scope reminder'):
  st.write(f'Analysis is for {ticker} prices from {period_start} to {period_end} with an interval of {interval} and moving average is based on {ma_period} days.')

# period1 = int(time.mktime(period_start))
# period2 = int(time.mktime(period_end))

# url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

# df = pd.read_csv(url)
# df['SMA'] = df['Close'].rolling(MA_period).mean()
# df['EMA'] = df['Close'].ewm(span=MA_period).mean()


# fig4 = go.Figure()
# fig4.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
#                     mode='lines',
#                     name='Close'))
# fig4.add_trace(go.Scatter(x=df["Date"], y=df["SMA"],
#                     mode='lines',
#                     name='SMA'))
# fig4.add_trace(go.Scatter(x=df["Date"], y=df["EMA"],
#                     mode='lines', name='EMA'))

# fig4.update_layout(
#     title=ticker+" Closing/SMA/EMA on "+str(MA_period)+" days",
#     autosize=False,
#     width=1500,
#     height=700)

# if plots == 'Prices' : st.plotly_chart(fig4)
