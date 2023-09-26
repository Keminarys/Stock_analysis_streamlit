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

def get_macd(df):
    df['Fast'] = df['Close'].ewm(span = 26, adjust = False).mean()
    df['Slow'] = df['Close'].ewm(span = 12, adjust = False).mean()
    df['MACD'] = df['Slow'] - df['Fast']
    df['Signal'] = df['MACD'].ewm(span = 9, adjust = False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def implement_macd_strategy(df):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(df)):
        if df['MACD'][i] > df['Signal'][i]:
            if signal != 1:
                buy_price.append(df['Close'][i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif df['MACD'][i] < df['Signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(df['Close'][i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal

def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price_bb = []
    sell_price_bb = []
    bb_signal = []
    signal = 0
    
    for i in range(1,len(data)):
        if data[i-1] > lower_bb[i-1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price_bb.append(data[i])
                sell_price_bb.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price_bb.append(np.nan)
                sell_price_bb.append(np.nan)
                bb_signal.append(0)
        elif data[i-1] < upper_bb[i-1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price_bb.append(np.nan)
                sell_price_bb.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price_bb.append(np.nan)
                sell_price_bb.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price_bb.append(np.nan)
            sell_price_bb.append(np.nan)
            bb_signal.append(0)
            
    return buy_price_bb, sell_price_bb, bb_signal

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
check_p= ['No','Yes']
check_v= ['No','Yes']
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
        period_start_i = st.date_input('Starting date for forecasting input data', datetime.datetime(2022,6,1))
        period_end_i = st.date_input('Ending date for forecasting input data', (datetime.date.today() + datetime.timedelta(days=1)))
        indic_to_plot = st.multiselect('Which indicator would you like to plot', pages_i)
        
with st.sidebar.expander("Portofolio Visualisation"):
    mult_ = st.radio('Would you like to plot more than one ticker ?', check_p)
    if mult_ == 'Yes' :
        st.write("""
        In order to visualize different stocks in one plot \n
        Please insert your tickers with a space (ex : AIR.PA ACA.PA)""")
        portfolio_ = st.text_input("Insert tickers here :point_down:")
        portfolio_ = portfolio_.split()

with st.sidebar.expander("PVariation Percentage Visualisation"):
    var_ = st.radio('Would you like to see variation in % ?', check_v)
    if var_ == 'Yes' :
        st.write("""
        In order to visualize different stocks in one plot \n
        Please insert your tickers with a space (ex : AIR.PA ACA.PA)""")
        variation_ = st.text_input("Insert tickers here :point_down:")
        variation_ = variation_.split()
    
st.write(f'Analysis is for {ticker} prices from {period_start} to {period_end} with an interval of {interval} and moving average is based on {ma_period} days.')
  
#####################################################
period1 = int(time.mktime(period_start.timetuple()))
period2 = int(time.mktime(period_end.timetuple()))

url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
df = pd.read_csv(url)
df_portfolio = pd.DataFrame()
df_variation =  pd.DataFrame()

ma_period_int = int(ma_period)
forecast_days_int = int(forecast_days)

df['SMA'] = df['Close'].rolling(ma_period_int).mean()
df['EMA'] = df['Close'].ewm(span=ma_period_int).mean()
indic = PSAR()

summary = pd.DataFrame(index=['Yesterday', 'Last Week', '52 Weeks', 'YTD'], columns=['Lowest', 'Highest', 'Variation'])

df['Date'] = pd.to_datetime(df['Date'])

yest = df.tail(1)[['Low', 'High']]
yest['var'] = round(((yest['High']-yest['Low'])/yest['Low']), 3)
summary.loc['Yesterday'] = yest.values

last_week = datetime.datetime.today() - datetime.timedelta(days=7)
last_week_data = df.loc[df['Date'] >= last_week]
lowtlw = last_week_data['Low'].min()
hightlw = last_week_data['High'].max()
vartlw = round(((hightlw-lowtlw)/lowtlw), 3)
summary.loc['Last Week'] = [lowtlw, hightlw, vartlw]

w52 = datetime.datetime.today() - datetime.timedelta(weeks=52)
w52_data = df.loc[df['Date'] >= w52]
lowt52 = w52_data['Low'].min()
hight52 = w52_data['High'].max()
vart52 = round(((hight52-lowt52)/lowt52), 3)
summary.loc['52 Weeks'] = [lowt52, hight52, vart52]

beg_ytd = datetime.datetime.today().replace(month=1, day=1)
ytd_data = df.loc[df['Date'] >= beg_ytd]
lowtytd = ytd_data['Low'].min()
hightytd = ytd_data['High'].max()
vartytd = round(((hightytd-lowtytd)/lowtytd), 3)
summary.loc['YTD'] = [lowtytd, hightytd, vartytd]

summary['Variation'] = summary['Variation'].apply(lambda x :"{0:.2f}%".format(x*100))
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
fig2.update_layout(width=1000, height=700, showlegend=False)

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
    width=1000,
    height=700)
########################################################

with st.container():
  col1, col2 = st.columns([4, 1])
  with col1 :
        if plots == 'Prices' : st.plotly_chart(fig4, use_container_width=True)
        if plots == 'Candlesticks' : st.plotly_chart(fig2, use_container_width=True)
  with col2 :
    st.dataframe(summary)  
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
    df_i['PSAR'] = df_i.apply(lambda x: indic.calcPSAR(x['High'], x['Low']), axis=1)
    df_i['EP'] = indic.ep_list
    df_i['Trend'] = indic.trend_list
    df_i['AF'] = indic.af_list
    psar_bull = df_i.loc[df_i['Trend']==1][['Date','PSAR']].set_index('Date')
    psar_bear = df_i.loc[df_i['Trend']==0][['Date','PSAR']].set_index('Date')
    buy_sigs = df_i.loc[df_i['Trend'].diff()==1][['Date','Close']].set_index('Date')
    short_sigs = df_i.loc[df_i['Trend'].diff()==-1][['Date','Close']].set_index('Date')
    df_i = get_macd(df_i)
    buy_price, sell_price, macd_signal = implement_macd_strategy(df_i)
    colors_bull_bear = ['springgreen', 'crimson', 'black']
    
    fig_indicators =  make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.02, subplot_titles=(ticker+" Candlestick",''), 
                row_width=[0.3, 0.7])
    fig_indicators.add_trace(go.Candlestick(x=df_i['Date'],
                open=df_i['Open'],
                high=df_i['High'],
                low=df_i['Low'],
                close=df_i['Close'], name = 'Price'), row=1, col=1)

    if 'MACD' in indic_to_plot:
        fig_indicators.add_trace(go.Scatter(x=df_i["Date"], y=df_i['MACD'],
                    mode='lines',
                    name='MACD', marker_color = 'grey'), row=2, col=1)
        fig_indicators.add_trace(go.Scatter(x=df_i["Date"], y=df_i['Signal'], mode='lines', name='Signal', marker_color = 'skyblue'), row=2, col=1)
        fig_indicators.add_trace(go.Bar(x=df_i['Date'], y=df_i['Hist'], showlegend=False, marker=dict(color=pd.Series(np.where(df_i['Hist'] < 0, 'red', 'green')
                       ).astype(str).map({'red':1, 'green':0}),colorscale=[[0, 'green'], [1, 'red']])), row=2, col=1)
        fig_indicators.add_trace(go.Scatter( x= df_i.Date, y = buy_price, name='Buy Signal MACD', mode='markers', marker_symbol='triangle-up-dot', marker_size=15, marker_color = 'blue'), row=1, col=1)
        fig_indicators.add_trace(go.Scatter( x= df_i.Date, y = sell_price, name='Short Signal MACD', mode='markers', marker_symbol='triangle-down-dot', marker_size=15, marker_color = 'blue'), row=1, col=1)

    if 'SAR' in indic_to_plot : 
        fig_indicators.add_trace(go.Scatter( x= psar_bull.index, y = psar_bull.PSAR, name='Up Trend', mode='markers', marker_color = colors_bull_bear[0]),row=1, col=1)
        fig_indicators.add_trace(go.Scatter( x= psar_bear.index, y = psar_bear.PSAR, name='Down Trend', mode='markers', marker_color = colors_bull_bear[1]),row=1, col=1)
        fig_indicators.add_trace(go.Scatter( x= buy_sigs.index, y = buy_sigs.Close, name='Buy Signal SAR', mode='markers', marker_symbol='triangle-up-dot', marker_size=15, marker_color = colors_bull_bear[2]),row=1, col=1)
        fig_indicators.add_trace(go.Scatter( x= short_sigs.index, y = short_sigs.Close, name='Sell Signal SAR', mode='markers', marker_symbol='triangle-down-dot', marker_size=15, marker_color = colors_bull_bear[2]), row=1, col=1)

    fig_indicators.update(layout_xaxis_rangeslider_visible=False)
    name_exp = str(indic_to_plot).strip('[]')
    fig_indicators.update_layout(title=ticker+" indicators : "+name_exp,autosize=False,width=2000,height=800)
    
    if 'Fibonacci' in indic_to_plot :
        
        highest_swing = lowest_swing = -1
        for i in range(1,df_i.shape[0]-1):
            if df_i['High'][i] > df_i['High'][i-1] and df_i['High'][i] > df_i['High'][i+1] and (highest_swing == -1 or df_i['High'][i] > df_i['High'][highest_swing]):
                highest_swing = i
            if df_i['Low'][i] < df_i['Low'][i-1] and df_i['Low'][i] < df_i['Low'][i+1] and (lowest_swing == -1 or df_i['Low'][i] < df_i['Low'][lowest_swing]):
                lowest_swing = i

        ratios = [0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
        colors = ["seagreen", "limegreen", "lightgreen","slategray","lightcoral", "firebrick", "darkred"]
        levels = []
        max_level = df_i['High'][highest_swing]
        min_level = df_i['Low'][lowest_swing]
        for ratio in ratios:
            if highest_swing > lowest_swing: # Uptrend
                levels.append(max_level - (max_level-min_level)*ratio)
            else: # Downtrend
                levels.append(min_level + (max_level-min_level)*ratio)
        for i in range(len(levels)):
            fig_indicators.add_hline(levels[i], line_dash="dot", line_color=colors[i],annotation_text=round(levels[i], 2), annotation_position="top right", annotation_font_size=10, annotation_font_color=colors[i], row=1)
      
    if 'Bollinger' in indic_to_plot : 
        
        std = df_i['Close'].rolling(window = 20).std()
        df_i['Upper_bb'] = df_i['Close'].rolling(20).mean() + std * 2
        df_i['Lower_bb'] = df_i['Close'].rolling(20).mean() - std * 2
        buy_price_bb, sell_price_bb, bb_signal = implement_bb_strategy(df_i['Close'], df_i['Lower_bb'], df_i['Upper_bb'])
        
        fig_indicators.add_trace(go.Scatter(x=df_i["Date"], y=df_i['Upper_bb'], mode='lines',  line = dict(shape = 'linear', color = 'rgb(70,130,180)', dash = 'dot'), name='Upper Bollinger Band'), row=1, col=1)
        fig_indicators.add_trace(go.Scatter(x=df_i["Date"], y=df_i['Lower_bb'], mode='lines',  line = dict(shape = 'linear', color = 'rgb(70,130,180)', dash = 'dot'), name='Lower Bollinger Band'), row=1, col=1)
        fig_indicators.add_trace(go.Scatter(x= df_i.Date, y = buy_price_bb, name='Buy Signal Bollinger', mode='markers', marker_symbol='triangle-up-dot', marker_size=15, marker_color = 'darkslateblue'), row=1, col=1)
        fig_indicators.add_trace(go.Scatter(x= df_i.Date, y = sell_price_bb, name='Short Signal Bollinger', mode='markers', marker_symbol='triangle-down-dot', marker_size=15, marker_color = 'darkslateblue'), row=1, col=1)
    with st.container():
        st.plotly_chart(fig_indicators)

#############################################################################################################
with st.container() :
    if len(portfolio_) > 0 and mult_ == "Yes":
        st.write("Performance of the chosen tickers")
        for i in portfolio_ :
            df_temp = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{i}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true')
            df_temp['Ticker'] = i
            df_portfolio = pd.concat([df_portfolio, df_temp], ignore_index=True)
        fig_port = px.line(df_portfolio, x="Date", y="Close", color='Ticker', log_y=True)
        st.plotly_chart(fig_port, use_container_width = True)

with st.container() :
    if len(variation_) > 0 and var_ == "Yes":
        st.write("Variation in % for chosen tickers")
        for i in variation_ :
            df_temp_var = pd.read_csv(f'https://query1.finance.yahoo.com/v7/finance/download/{i}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true')
            df_temp_var['Ticker'] = i
            df_variation = pd.concat([df_variation, df_temp_var], ignore_index=True)
            
        fig_var = make_subplots(rows=len(df_variation.Ticker.unique()), cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(df_variation.Ticker.unique()))

        for i, r, c in zip(df_variation.Ticker.unique(), range(1,len(df_variation.Ticker.unique())+1,1), ['rgb(120, 0, 0)', 'rgb(0, 120, 0)', 'rgb(0, 0, 120)']) :
              
            df_portfolio_temp = df_variation.loc[df_variation['Ticker'] == i]

            fig_var.add_trace(go.Scatter(name=i,
                                   x=df_portfolio_temp['Date'],
                                   y=df_portfolio_temp['Open_Close_%'],
                                   mode='lines',
                                   line=dict(color=c),
                                  ), row=r, col=1)
            fig_var.add_trace(go.Scatter(
                                    name='High %',
                                    x=df_portfolio_temp['Date'],
                                    y=df_portfolio_temp['High_%'],
                                    mode='lines',
                                    marker=dict(color="#444"),
                                    line=dict(width=0),
                                    fillcolor='rgba(68, 68, 68, 0.3)',
                                    showlegend=False
                                    ), row=r, col=1)
            fig_var.add_trace(go.Scatter(
                                    name='Low %',
                                    x=df_portfolio_temp['Date'],
                                    y=df_portfolio_temp['Low_%'],
                                    marker=dict(color="#444"),
                                    line=dict(width=0),
                                    mode='lines',
                                    fillcolor='rgba(68, 68, 68, 0.3)',
                                    fill='tonexty',
                                    showlegend=False
                                    ),row=r, col=1)
            fig_var.update_yaxes(title_text="Variation in %", row=r, col=1)
            fig_var.update_layout(
                            title='Variation in % with lowest and highest weekly',
                            hovermode="x",
                            height=800,
                            width=1200
                            )
        st.plotly_chart(fig_var, use_container_width = True)
