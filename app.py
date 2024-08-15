import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Load the model
model = load_model('C:\\Users\\kabir\\OneDrive\\Desktop\\STOCK\\Stock Predictions Model.keras')

# Streamlit Header
st.header('Stock Market Prediction')

# User Input for Stock Ticker
stock = st.text_input('Enter Ticker symbol', 'RELIANCE.NS')

# Dates for historical data
start = '1996-1-1'
end = '2024-08-12'
data = yf.download(stock, start, end)

# Display Stock Data
st.subheader('Stock Data')
st.write(data)

# Prepare data for training/testing
data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days,data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Moving Averages
st.subheader('Price vs MA50')
ma_50_days = data['Close'].rolling(50).mean()
fig1 = plt.figure(figsize=(10,10))
plt.plot(ma_50_days,'r')
plt.plot(data['Close'], 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data['Close'].rolling(100).mean()
fig2 = plt.figure(figsize=(10,10))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data['Close'], 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA50 vs MA200')
ma_200_days = data['Close'].rolling(200).mean()
fig3 = plt.figure(figsize=(10,10))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(ma_200_days,'black')
plt.plot(data['Close'], 'g')
plt.show()
st.pyplot(fig3)

# Prepare data for model prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y = y*scale

# Plot Original vs Predicted Prices
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,10))
plt.plot(predict,'r',label='Predicted Price')
plt.plot(y,'g',label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)

# # User Input for Date
# st.subheader('Predict Stock Price for a Specific Date')
# date_input = st.date_input('Select a date', value=datetime(2024, 8, 13))
# date_str = date_input.strftime('%Y-%m-%d')

# # Prepare data for the selected date
# if date_str in data.index:
#     past_100_days = data.loc[:date_str].tail(100)[['Close']]  # Only use the 'Close' column
#     past_100_days_scaled = scaler.transform(past_100_days)

#     x_input = np.array([past_100_days_scaled])

#     # Predict the price for the selected date
#     predicted_price = model.predict(x_input)
#     predicted_price = predicted_price * scale


#     st.write(f'Predicted price for {date_str}: ₹{predicted_price[0][0]:.2f}')
# else:
#     st.write(f'Data not available for {date_str}. Please select a different date.')


