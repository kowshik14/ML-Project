import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as dataread
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, SpatialDropout1D, Embedding, LSTM, GRU, Bidirectional, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import Dense, Dropout, Activation, Embedding
import datetime
import seaborn as sb
import yfinance as yf
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


start= '2010-01-01'
end=datetime.date.today()

st.title('Stock Price Forecasting')

user_input=st.text_input('Enter Stock Tiker', 'CSCO')
dataset=yf.download(user_input, start, end)

#Describe
st.subheader('Data from 2010 to Today')
st.write(dataset.describe())

#visualization of EDA

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize =(12,6))
plt.xlabel('Time')
plt.ylabel('Close Price USD ($)')
plt.plot(dataset.Close)
st.pyplot(fig)



ma100=dataset.Close.rolling(100).mean()
ma200=dataset.Close.rolling(200).mean()
st.subheader('Closing Price vs Time Chart with Moving Average')
fig2=plt.figure(figsize=(12,6))
plt.plot(dataset.Close)
plt.plot(ma100,'r', label= 'MA100')
plt.plot(ma200,'g', label= 'MA200')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Close Price USD ($)')
plt.show()
st.pyplot(fig2)



st.subheader('Distribuition Plot')
features = ['Open', 'High', 'Low', 'Close', 'Volume']
fig3,ax=plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  ax.hist(dataset[col])
  sb.distplot(dataset[col])
st.pyplot(fig3)
st.write('In the distribution plot of OHLC data, we can see two peaks which means the data has varied significantly in two regions. And the Volume data is left-skewed.')


st.subheader('Box Plot')
fig4,ax4=plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  ax4.hist(dataset[col])
  sb.boxplot(dataset[col])
st.pyplot(fig4)
st.write('From the above boxplots, we can conclude that only volume data contains outliers in it but the data in the rest of the columns are free from any outlier.')

dataset=dataset.reset_index()
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['day'] = dataset['Date'].dt.day
dataset['month'] = dataset['Date'].dt.month
dataset['year'] = dataset['Date'].dt.year


data_grouped = dataset.groupby('year').mean()
st.subheader('Bar Chart')
fig5,ax5=plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
  plt.subplot(2,2,i+1)
  data_grouped[col].plot.bar()
  ax5.hist(data_grouped[col])
st.pyplot(fig5)
st.write('From the above bar graph, we can conclude that the stock prices have increased consecutively from the year 2011 to that in 2019.')


st.subheader('Groped Data')
dataset['is_quarter_end'] = np.where(dataset['month']%3==0,1,0)
st.write(dataset.groupby('is_quarter_end').mean())
st.write('Here are some of the important observations of the above-grouped data:\n -Prices are almost same in the months which are quarter end as compared to that of the non-quarter end months.\n The volume of trades is very lower in the months which are quarter end.')


dataset['open-close']  = dataset['Open'] - dataset['Close']
dataset['low-high']  = dataset['Low'] - dataset['High']
dataset['target'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)


st.subheader('Pie Chart')
fig6=plt.figure(figsize =(12,6))
plt.pie(dataset['target'].value_counts().values, labels=[0, 1], autopct='%1.1f%%')
st.pyplot(fig6)
st.write('When we add features to our dataset we have to ensure that there are no highly correlated features as they do not help in the learning process of the algorithm.')


st.subheader('Correlation')
fig7=plt.figure(figsize=(10, 10)) 
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(dataset.corr() > 0.9, annot=True, cbar=False)
st.pyplot(fig7)
st.write('From the above heatmap, we can say that there is a high correlation between OHLC that is pretty obvious, and the added features are not highly correlated (except "year") with each other or previously provided features which means that we are good to go and build our model.')

time_step=100


#splitting
train_data=pd.DataFrame(dataset['Close'][0:int(len(dataset)*.70)])
test_data=pd.DataFrame(dataset['Close'][int(len(dataset)*.70):int(len(dataset))])


past_time_index=train_data.tail(time_step)
final_test_data=past_time_index.append(test_data,ignore_index=True)

#scaling
scaler=MinMaxScaler(feature_range=(0,1))

training_data=scaler.fit_transform(train_data)
testing_data=scaler.fit_transform(final_test_data)


# convert an array of values into a dataset matrix
def create_dataset(dataset,time_step):
    dataX = []
    dataY = []
    time_step=100
    for i in range(time_step, dataset.shape[0]):
        dataX.append(dataset[i-time_step: i])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)

x_train, y_train = create_dataset(training_data,100)
x_test, y_test = create_dataset(testing_data,100)


#load model
model=load_model('model.h5',compile=False)
model.compile(optimizer='adam', loss="mean_squared_error")




y_pred=model.predict(x_test)

scale_factor=1/scaler.scale_
y_pred1=y_pred * scale_factor
y_test1=y_test * scale_factor



st.subheader('Actual vs Predicted Closing Price Graph')
fig10=plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(y_test1, 'g', label ='Original Price')
plt.plot(y_pred1,'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='best')
st.pyplot(fig10)


st.subheader('Actual vs Predicted Closing Price')
test_data['Prediction']=scaler.inverse_transform(y_pred)
st.write(test_data[['Close','Prediction']])
