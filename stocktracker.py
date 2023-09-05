import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


#start & end dates
START = "2017-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.write("""
# Welcome to the Tech Stock Price Tracker/Projector

#### Kevin Lau


----

""")

#tech stocks to display
stocks = ('GOOGL', 'AAPL', 'MSFT', 'META', 'TSLA', 'NVDA')
selected_stock = st.selectbox('Select stock', stocks)

#period of forecast
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

#cache stock data
@st.cache_data
#load stock data from library
def load_data(ticker):
    data = yf.download(ticker, period='60mo')
#align start and end date 
    data.reset_index(inplace=True)
    return data

#load raw data 	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

#formatting raw data chart
st.subheader('Raw data')
st.write(data.tail())

# plot historical stock data function
def plot_raw_data():
	#plotly graph figure
	fig = go.Figure()
	#x and y data + axis of graph 
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open Price"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Price"))
	fig.layout.update(title_text='Closing Price Data', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
#call func to create graph 
plot_raw_data()


#forecasting ---------------------

#format dataframe of the stock 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

#generate forecast over period using fbprohet forecasting library
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# display forecast chart
st.subheader('Forecast data')
st.write(forecast.tail())
    
#plot forecast graph 
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
