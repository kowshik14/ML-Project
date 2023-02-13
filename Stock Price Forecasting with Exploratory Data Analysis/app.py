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
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


start= '2010-01-01'
end=datetime.date.today()

st.title('Stock Price Forecasting')

user_input=st.text_input('Enter Stock Tiker', 'AAPL')
dataset=dataread.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from 2010 to Today')
st.write(dataset.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize =(12,6))
plt.plot(dataset.Close)
st.pyplot(fig)

ma7=dataset.Close.rolling(7).mean()
ma21=dataset.Close.rolling(21).mean()
ma60=dataset.Close.rolling(60).mean()
st.subheader('Moving average of 7, 21, 60 days')
fig2=plt.figure(figsize=(12,6))
plt.plot(dataset.Close)
plt.plot(ma7,'r')
plt.plot(ma21,'g')
plt.plot(ma60)
plt.show()
st.pyplot(fig2)

# Create a new dataframe with only the 'Close column 
data=dataset.filter(['Close'])
#convert dataset to numpy array
df=data.values

#standardization
scaler=StandardScaler()
scaler_data=scaler.fit_transform(df)

# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(df) * .95 ))
window_size=60

# Create the training data set 
# Create the scaled training data set

train_data = scaler_data[0:int(training_data_len), :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
    if i<= window_size+1:
        print(x_train)
        print(y_train)
        print()
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model=load_model('model2.h5')

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaler_data[training_data_len - window_size: ]
print ('len(test_data):', len(test_data))

# Create the data sets x_test and y_test
x_test = []
y_test = scaler_data[training_data_len:, :]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#prediction
pred=model.predict(x_test)
pred=scaler.inverse_transform(pred)

# Plot the data
train = dataset[:training_data_len]
test = dataset[training_data_len:]
test['Predictions'] = pred

st.subheader('Actual vs Predicted Closing Price Graph')
# Visualize the data
fig3=plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
#plt.plot(train['Close'])
#plt.plot(test[['Close', 'Predictions']])
plt.plot(test.Close)
plt.plot(test.Predictions,'r')
plt.legend(['Close', 'Predictions'], loc='best')
plt.show()
st.pyplot(fig3)

st.subheader('Actual vs Predicted Closing Price')
st.write(test[['Close','Predictions']])