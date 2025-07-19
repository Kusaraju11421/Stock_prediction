import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model

import streamlit as st
import matplotlib.pyplot as plt
import base64
import io
st.set_page_config(layout="wide") #interface layout

#applying stylings for buttons and containers
st.markdown(
    """
    <style>

    .blur-container {
        background: rgba(255, 255, 255, 0.05); 
        backdrop-filter: blur(15px); 
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.5);
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTextInput, .stNumberInput, .stSelectbox {
        color: white !important;
        background: rgba(0, 0, 255, 0.1) !important; 
        backdrop-filter: blur(10px) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 0, 255, 0.1) !important;
        padding: 8px !important;
    }

   
    .stButton>button {
        background: rgba(0, 255, 0, 0.0); 
        backdrop-filter: blur(10px);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
        border: 1px solid rgba(0, 0, 0, 0.0);
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: rgba(0, 0, 0, 0.0);
    }
    .stApp {
        padding: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#setting custom background 
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encode = base64.b64encode(image_file.read()).decode()
    background = f"""
    <style>
    .stApp{{
    background-image:url("data:image/png;base64,{encode}");
    background-size:cover;
    background-position:center;
    background-repeat:no-repeat;
    }}
    </style>
    """
    st.markdown(background, unsafe_allow_html=True)

#reading background image
set_background(r"C:\Users\kusar\OneDrive\Pictures\sasuke.jpeg")




model = load_model(r'K:\mine_infoooo\corizo\stock_prediction\Stock_Prediction_Model.keras')

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
# import matplotlib.pyplot as plt
# import streamlit as st

col1,col2,col3 = st.columns(3)
with col1:
    st.subheader("         "+"  Price vs Moving Average of 50 Days")

    mv_avg_50 = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(9, 5))  # Smaller canvas
    plt.plot(mv_avg_50, 'black', label='Moving Average of 50 Days')
    plt.plot(data.Close, 'purple', label='Closing Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=800)  # Adjust width to reduce visual size


with col2:
    st.subheader("Price vs Moving Average of 50 vs  100 Days")
    mv_avg_100 = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(9,5))
    plt.plot(mv_avg_50,'black',label = 'Moving Average of 50 Days')
    plt.plot(mv_avg_100,'blue',label = 'Moving Average of 100 Days')
    plt.plot(data.Close,'red',label = 'Closing Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=800) 
with col3:
    st.subheader("Price vs Moving Average of 100  vs 200 Days")
    mv_avg_200 = data.Close.rolling(200).mean()
    fig3= plt.figure(figsize=(9,5))
    plt.plot(mv_avg_100,'black',label = 'Moving Average of 50 Days')
    plt.plot(mv_avg_200,'blue',label = 'Moving Average of 100 Days')
    plt.plot(data.Close,'green',label = 'Closing Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.image(buf, width=800) 

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


st.markdown("<h3 style='text-align: center;'>Original Price vs Predicted Price</h3>", unsafe_allow_html=True)

# Create the plot
fig4 = plt.figure(figsize=(9, 5))
plt.plot(predict, 'black', label='Original Price')
plt.plot(y, 'blue', label='Predicted Price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

# Save the plot to a buffer
buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)
plt.close(fig4)

# Center the image using a middle column
col1, col2, col3 = st.columns([1, 2, 1])  # 3-column layout for centering
with col2:
    st.image(buf, use_container_width=True)
