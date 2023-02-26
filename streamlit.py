import pandas as pd
import numpy as np
import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import streamlit as st
from collections import deque

def date_range(start, end):
    delta = end - start
    days = [start + datetime.timedelta(days=i) for i in range(delta.days + 1)]
    return days

class PSAR:

  def __init__(self, init_af=0.02, max_af=0.2, af_step=0.02):
    self.max_af = max_af
    self.init_af = init_af
    self.af = init_af
    self.af_step = af_step
    self.extreme_point = None
    self.high_price_trend = []
    self.low_price_trend = []
    self.high_price_window = deque(maxlen=2)
    self.low_price_window = deque(maxlen=2)

    # Lists to track results
    self.psar_list = []
    self.af_list = []
    self.ep_list = []
    self.high_list = []
    self.low_list = []
    self.trend_list = []
    self._num_days = 0

  def calcPSAR(self, high, low):
    if self._num_days >= 3:
      psar = self._calcPSAR()
    else:
      psar = self._initPSARVals(high, low)

    psar = self._updateCurrentVals(psar, high, low)
    self._num_days += 1

    return psar

  def _initPSARVals(self, high, low):
    if len(self.low_price_window) <= 1:
      self.trend = None
      self.extreme_point = high
      return None

    if self.high_price_window[0] < self.high_price_window[1]:
      self.trend = 1
      psar = min(self.low_price_window)
      self.extreme_point = max(self.high_price_window)
    else: 
      self.trend = 0
      psar = max(self.high_price_window)
      self.extreme_point = min(self.low_price_window)

    return psar

  def _calcPSAR(self):
    prev_psar = self.psar_list[-1]
    if self.trend == 1: # Up
      psar = prev_psar + self.af * (self.extreme_point - prev_psar)
      psar = min(psar, min(self.low_price_window))
    else:
      psar = prev_psar - self.af * (prev_psar - self.extreme_point)
      psar = max(psar, max(self.high_price_window))

    return psar

  def _updateCurrentVals(self, psar, high, low):
    if self.trend == 1:
      self.high_price_trend.append(high)
    elif self.trend == 0:
      self.low_price_trend.append(low)

    psar = self._trendReversal(psar, high, low)

    self.psar_list.append(psar)
    self.af_list.append(self.af)
    self.ep_list.append(self.extreme_point)
    self.high_list.append(high)
    self.low_list.append(low)
    self.high_price_window.append(high)
    self.low_price_window.append(low)
    self.trend_list.append(self.trend)

    return psar

  def _trendReversal(self, psar, high, low):
    # Checks for reversals
    reversal = False
    if self.trend == 1 and psar > low:
      self.trend = 0
      psar = max(self.high_price_trend)
      self.extreme_point = low
      reversal = True
    elif self.trend == 0 and psar < high:
      self.trend = 1
      psar = min(self.low_price_trend)
      self.extreme_point = high
      reversal = True

    if reversal:
      self.af = self.init_af
      self.high_price_trend.clear()
      self.low_price_trend.clear()
    else:
        if high > self.extreme_point and self.trend == 1:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = high
        elif low < self.extreme_point and self.trend == 0:
          self.af = min(self.af + self.af_step, self.max_af)
          self.extreme_point = low

    return psar

st.set_page_config(layout="wide")

st.title('Stock analyzer')  

pages = ['Prices','Candlesticks']
pages_f = ['No','Yes']
check_i= ['No','Yes']
pages_i = ['MACD', 'SAR', 'Bollinger', 'Fibonacci', 'Stochastics oscillator']

with st.sidebar.expander("General Input"):
  st.write("This section is used for stock analysis")
  ticker = st.text_input('Please enter a ticker', "AIR.PA")
  period_start = st.date_input('Please enter starting date', datetime.datetime(2021,1,1))
  period_end = st.date_input('Please enter ending date', (datetime.date.today() + datetime.timedelta(days=1)))
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

with st.sidebar.expander("Technical Analysis Indicator"):
    more_opt = st.radio('Would you like to plot some indicators ?', check_i)
    if more_opt == 'Yes' :
        st.write("You can choose different key indicators here")
        period_start_i = st.date_input('Starting date for forecasting input data', datetime.datetime(2023,1,1))
        period_end_i = st.date_input('Ending date for forecasting input data', datetime.date.today() + datetime.timedelta(days=1))
        indic_to_plot = st.multiselect('Which indicator would you like to plot', pages_i)


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
indic = PSAR()

df['PSAR'] = df.apply(lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
df['EP'] = indic.ep_list
df['Trend'] = indic.trend_list
df['AF'] = indic.af_list
#######################################################

### Candlesticks
fig2 =  make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.02, subplot_titles=(ticker+" Candlestick",''), 
               row_width=[0.2, 0.7])
fig2.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"]), 
                row=1, col=1)
fig2.add_trace(go.Bar(x=df['Date'], y=df['Volume'], showlegend=False), row=2, col=1)
fig2.update(layout_xaxis_rangeslider_visible=False)
fig2.update_layout(width=1500, height=700, showlegend=False)

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
    
#######################################################
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
if more_opt == 'Yes' :
    period1_i = int(time.mktime(period_start_i.timetuple()))
    period2_i = int(time.mktime(period_end_i.timetuple()))
    url_i = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1_i}&period2={period2_i}&interval={interval}&events=history&includeAdjustedClose=true'
    df_i = pd.read_csv(url_i)
    psar_bull = df_i.loc[df_i['Trend']==1][['Date','PSAR']].set_index('Date')
    psar_bear = df_i.loc[df_i['Trend']==0][['Date','PSAR']].set_index('Date')
    buy_sigs = df_i.loc[df_i['Trend'].diff()==1][['Date','Close']].set_index('Date')
    short_sigs = df_i.loc[df_i['Trend'].diff()==-1][['Date','Close']].set_index('Date')
    if "SAR" in indic_to_plot :
        colors_bull_bear = ['springgreen', 'crimson', 'black']
        fig_sar = go.Figure([go.Candlestick(x=df_i['Date'],
                open=df_i['Open'],
                high=df_i['High'],
                low=df_i['Low'],
                close=df_i['Close'], name = 'Price'),
        go.Scatter( x= psar_bull.index, y = psar_bull.PSAR, name='Up Trend', mode='markers', marker_color = colors_bull_bear[0]), 
        go.Scatter( x= psar_bear.index, y = psar_bear.PSAR, name='Down Trend', mode='markers', marker_color = colors_bull_bear[1]), 
        go.Scatter( x= buy_sigs.index, y = buy_sigs.Close, name='Buy', mode='markers', marker_symbol='triangle-up-dot', marker_size=15, marker_color = colors_bull_bear[2]),
        go.Scatter( x= short_sigs.index, y = short_sigs.Close, name='Short', mode='markers', marker_symbol='triangle-down-dot', marker_size=15, marker_color = colors_bull_bear[2])])
    
        fig_sar.update(layout_xaxis_rangeslider_visible=False)
        fig_sar.update_layout(title=ticker+" SAR indicator",autosize=False,width=1500,height=700)
        with st.container():
            st.plotly_chart(fig_sar)
