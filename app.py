import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'K:\mine_infoooo\corizo\stock_prediction\Stock_Prediction Model.keras')

st.header("Stock Market Predictor")

stock = st.text_input('Enter Stock Symbol','TATAMOTORS.NS')
start = '2020-01-01'
end = '2024-12-31'

data = yf.download(stock,start,end)

st.subheader("Stock Data")
st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days,data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader("Price vs Moving Average of 50 Days")
mv_avg_50 = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(mv_avg_50,'black',label = 'Moving Average of 50 Days')
plt.plot(data.Close,'purple',label = 'Closing Price')
plt.legend()
plt.show()
st.pyplot(fig1)


st.subheader("Price vs Moving Average of 50 Days vs  100 Days")
mv_avg_100 = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(mv_avg_50,'black',label = 'Moving Average of 50 Days')
plt.plot(mv_avg_100,'blue',label = 'Moving Average of 100 Days')
plt.plot(data.Close,'red',label = 'Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)


st.subheader("Price vs Moving Average of 100 Days vs 200 Days")
mv_avg_200 = data.Close.rolling(200).mean()
fig3= plt.figure(figsize=(8,6))
plt.plot(mv_avg_100,'black',label = 'Moving Average of 50 Days')
plt.plot(mv_avg_200,'blue',label = 'Moving Average of 100 Days')
plt.plot(data.Close,'green',label = 'Closing Price')
plt.legend()
plt.show()
st.pyplot(fig3)


st.write(data)



x=[]
y=[]
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict*scale
y = y * scale

st.subheader("Original Price vs Predicted Price")
fig4= plt.figure(figsize=(8,6))
plt.plot(predict,'black',label = 'Original Price')
plt.plot(y,'blue',label = 'Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig4)
