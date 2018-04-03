#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:04:14 2018

@author: richard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#Get the open value
training_set = dataset_train.iloc[:,1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range= (0,1))
training_set = sc.fit_transform (training_set)

#Creating the a real data to be processed
X_train = []
y_train = []
for i in range(60, 1258): # don't include the last one
    X_train.append(training_set[i-60:i,:])
    y_train.append(training_set[i,:])
X_train, y_train = np.array(X_train), np.array(y_train)

#X_train = np.reshape(X_train,(X_train.shape[0], X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential() # cuz we are analysising a contineous value -> regressor

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(1))

regressor.compile(optimizer ='adam', loss ='mean_squared_error')

regressor.fit(X_train,y_train, epochs= 100, batch_size=32)


dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values  # get the real price


# combining 2 data together

total_dataset = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)
inputs = total_dataset[len(total_dataset)- len(dataset_test)-60:].values
inputs = inputs.reshape(-1,1)  # 1D -> 2D
inputs = sc.transform(inputs) # scaling before prediction
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60:i,:])
X_test = np.array(X_test)
#X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform (predicted_stock_price)

#visualising 

plt.plot(real_stock_price, color ='red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color ='blue', label = 'Predicted Stock Price')
plt.title("Google stock price")
plt.xlabel("Time")
plt.ylabel("Google stock price")
plt.legend()
plt.show()





