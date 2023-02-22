import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import streamlit as st

def date_range(start, end):
    delta = end - start
    days = [start + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    return days

st.set_page_config(layout="wide")

st.title('Stock analyzer')  

pages = ['Prices','Candlesticks','Volume']
pages_f = ['No','Yes']

with st.sidebar.expander("General Input"):
  st.write("This section is used for stock analysis")
  ticker = st.text_input('Please enter a ticker', "AIR.PA")
  period_start = st.date_input('Please enter starting date', datetime.datetime(2021,1,1))
  period_end = st.date_input('Please enter ending date', (datetime.date.today() - datetime.timedelta(days=1)))
  ma_period = st.text_input('Please enter a moving average period', "50")
  interval = st.selectbox('Please choose an interval', ['1d', '1wk', '1mo'])
  plots = st.radio('Select a plot to show', pages)
 
with st.sidebar.expander("Forecast Input"):
  st.write("""
  This section is used for forecasting \n
  Note that if you want to forecast for a long period of time you will need a high year delta.
  """)
  period_start_f = st.date_input('Starting date for forecasting input data', datetime.datetime(2021,1,1))
  period_end_f = st.date_input('Ending date for forecasting input data', datetime.datetime(2023,1,1))
  forecast_days = st.text_input('Number of days used to forecast', "15")
  plots_f = st.radio('Show forecast plot ?', pages_f)

st.write(f'Analysis is for {ticker} prices from {period_start} to {period_end} with an interval of {interval} and moving average is based on {ma_period} days.')
  
#####################################################
period1 = int(time.mktime(period_start.timetuple()))
period2 = int(time.mktime(period_end.timetuple()))

url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'

df = pd.read_csv(url)

ma_period_int = int(ma_period)
forecast_days_int = int(forecast_days)

df['SMA'] = df['Close'].rolling(ma_period_int).mean()
df['EMA'] = df['Close'].ewm(span=ma_period_int).mean()
#######################################################

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
########################################################

with st.container():
  if plots == 'Prices' : st.plotly_chart(fig4)
  if plots == 'Candlesticks' : st.plotly_chart(fig2)
  if plots == 'Volume' : st.plotly_chart(fig3)
    
if plots_f == 'Yes' : 
  
  period1_f = int(time.mktime(period_start_f.timetuple()))
  period2_f = int(time.mktime(period_end_f.timetuple()))
  url_f = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1_f}&period2={period2_f}&interval={interval}&events=history&includeAdjustedClose=true'
  df_f = pd.read_csv(url_f)

  startDatec = datetime.datetime(2020, 2, 17)
  endDatec = datetime.datetime(2020, 3, 17)      
  datescovid = date_range(startDatec, endDatec)
  startDatew = datetime.datetime(2022, 2, 24)
  endDatew = datetime.datetime(2022, 3, 10)      
  dateswar = date_range(startDatew, endDatew)

  covid = pd.DataFrame({
    'holiday': 'covid',
    'ds': datescovid,
    'lower_window': 0,
    'upper_window': 1,})
  war = pd.DataFrame({
    'holiday': 'war',
    'ds': dateswar,
    'lower_window': 0,
    'upper_window': 1,})
  events = pd.concat((covid, war))

  df_proph = df_f[['Date', 'Close']]
  df_proph.columns = ['ds', 'y']
  m = Prophet(daily_seasonality=False, weekly_seasonality=False, holidays = events)
  m.fit(df_proph)
  future = m.make_future_dataframe(periods=forecast_days_int)
  forecast = m.predict(future)
  with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.header("Past and Forecast using Prophet")
        st.plotly_chart(plot_plotly(m, forecast))
    with col2:
        st.header("Trends")
        st.plotly_chart(plot_components_plotly(m, forecast))
############################################################
