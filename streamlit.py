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
