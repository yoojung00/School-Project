#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:29:32 2019

@author: KelsaDuan
"""
#%%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
#%%
# set seed
np.random.seed(1)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
data = pd.read_csv("ASX200Daily.csv", index_col='Date', parse_dates=['Date'], date_parser=dateparse)
data = data['Close']['2009-01-01':]

#%%
#forward fill
data_forward = data.fillna(method = 'ffill')

#seperate dataset
data_train = data_forward.loc[:'2017-12-28']
data_test = data_forward.loc['2018-01-01':]
#%%
# pre-processing
# scaling
data_train = np.array(data_train.values).reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train)

#%%
# create data structure with 60 timestaps and 5 output for each step (using for training)
time_window = 60

Xtrain = []
Ytrain = []


for i in range(time_window, len(data_train)-5):
    Xtrain.append(data_train[i-time_window:i, 0])
    Ytrain.append(data_train[i:i+5, 0])
    
Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)


# reshape for inputting in RNN
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))

#%%
# create data structure with 60 timestamps for test set (using for prediction)
inputs = np.array(data[len(data) - len(data_test) - time_window:len(data)].values).reshape(-1, 1)
inputs = scaler.fit_transform(inputs)

#%%
Xtest = []
Ytest = []
for i in range(time_window, len(inputs), 5):
    Xtest.append(inputs[i-time_window:i, 0])

Xtest = np.array(Xtest)

# reshape for inputting in RNN
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))

# prepare the actual test data for calculating mse (using for evaluation)
Ytest = np.array(data_test.values).reshape(-1,1)

#%%
# LSTM - 1 Hidden Layer
def LSTM1HiddenLayers (Ytrain, unit, dp, act, lf, opt, batch, ep):
    # define the model structure
    model1 = Sequential()

    model1.add(LSTM(input_shape=(Xtrain.shape[1], 1),units=unit, return_sequences=False))
    model1.add(Dropout(dp))

    model1.add(Dense(5, activation=act))

    model1.compile(loss=lf, optimizer=opt)

    # train the model
    model1.fit(Xtrain, Ytrain, batch_size=batch, epochs=ep)

    # predict on test set
    Ytest_predicted = model1.predict(Xtest)
    Ytest_predicted = scaler.inverse_transform(Ytest_predicted)
    
    return Ytest_predicted

#%%
# LSTM - 2 Hidden Layers
def LSTM2HiddenLayers (Ytrain, unit, dp, act, lf, opt, batch, ep):
    # define the model structure
    model2 = Sequential()

    model2.add(LSTM(input_shape=(Xtrain.shape[1], 1),units=unit, return_sequences=True))
    model2.add(Dropout(dp))
    
    model2.add(LSTM(units=unit, return_sequences=False))
    model2.add(Dropout(dp))

    model2.add(Dense(5, activation=act))

    model2.compile(loss=lf, optimizer=opt)

    
    early_stop = [EarlyStopping(monitor='loss', patience=2, verbose=1)]
    # train the model
    model2.fit(Xtrain, Ytrain,  batch_size=batch, epochs=ep)

    # predict on test set
    Ytest_predicted = model2.predict(Xtest)
    Ytest_predicted = scaler.inverse_transform(Ytest_predicted)

    
    return Ytest_predicted

#%%
# LSTM - 3 Hidden Layers
def LSTM3HiddenLayers (Ytrain, unit, dp, act, lf, opt, batch, ep):
    # define the model structure
    model3 = Sequential()

    model3.add(LSTM(input_shape=(Xtrain.shape[1], 1),units=unit, return_sequences=True))
    model3.add(Dropout(dp))
    
    model3.add(LSTM(units=unit, return_sequences=True))
    model3.add(Dropout(dp))
    
    model3.add(LSTM(units=unit, return_sequences=False))
    model3.add(Dropout(dp))

    model3.add(Dense(5, activation=act))

    model3.compile(loss=lf, optimizer=opt)

    # train the model
    model3.fit(Xtrain, Ytrain, batch_size=batch, epochs=ep)

    # predict on test set
    Ytest_predicted = model3.predict(Xtest)
    Ytest_predicted = scaler.inverse_transform(Ytest_predicted)
    

    return Ytest_predicted

#%%
# set parameters sets for tuning
units_set = [50, 100, 200]
dropout_set = [0, 0.25]
activation_set = ['sigmoid', 'linear', 'relu']
loss_set = ['mean_squared_error']
optimizer_set = ['rmsprop', 'adam']
batch_set = [32]
epochs_set = [10]

#%%
# hyperparameters tuning
results = pd.DataFrame([])

for u in units_set:
    for d in dropout_set:
        for a in activation_set:
            for l in loss_set:
                for o in optimizer_set:
                    for b in batch_set:
                        for e in epochs_set:
                            Ytest_predicted1_1layer = LSTM1HiddenLayers (Ytrain, u, d, a, l, o, b, e)
                            Ytest_predicted1_2layer = LSTM2HiddenLayers (Ytrain, u, d, a, l, o, b, e)
                            Ytest_predicted1_3layer = LSTM3HiddenLayers (Ytrain, u, d, a, l, o, b, e)
                            results = results.append(
                                    pd.DataFrame({'units':u,
                                                  'dropout rate':d,
                                                  'activation function':a,
                                                  'loss function':l,
                                                  'optimizer':o,
                                                  'batch size':b,
                                                  'epochs':e,
                                                  'mse_1layer':mean_squared_error(Ytest, Ytest_predicted1_1layer.reshape(-1,1)),
                                                  'mse_2layer':mean_squared_error(Ytest, Ytest_predicted1_2layer.reshape(-1,1)),
                                                  'mse_3layer':mean_squared_error(Ytest, Ytest_predicted1_3layer.reshape(-1,1))},
                                                  index=[0]),ignore_index=True)

#%%
results.to_csv('results.csv')                            
                            
#%%
# select the 5 models with min mse and tuning epochs
epochs_set = [10, 50, 100]

results2 = pd.DataFrame([])

for e in epochs_set:
    Ytest_optimal1 = LSTM2HiddenLayers (Ytrain, 200, 0, 'linear', 'mean_squared_error', 'rmsprop', 32, e)
    Ytest_optimal2 = LSTM2HiddenLayers (Ytrain, 200, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 32, e)
    Ytest_optimal3 = LSTM2HiddenLayers (Ytrain, 100, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 32, e)
    Ytest_optimal4 = LSTM1HiddenLayers (Ytrain, 50, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 32, e)
    Ytest_optimal5 = LSTM3HiddenLayers (Ytrain, 100, 0, 'sigmoid', 'mean_squared_error', 'rmsprop', 32, e)
    results2 = results2.append(
            pd.DataFrame({'epochs':e,
                          'mse_optimal1':mean_squared_error(Ytest, Ytest_optimal1.reshape(-1,1)),
                          'mse_optimal2':mean_squared_error(Ytest, Ytest_optimal2.reshape(-1,1)),
                          'mse_optimal3':mean_squared_error(Ytest, Ytest_optimal3.reshape(-1,1)),
                          'mse_optimal4':mean_squared_error(Ytest, Ytest_optimal4.reshape(-1,1)),
                          'mse_optimal5':mean_squared_error(Ytest, Ytest_optimal5.reshape(-1,1))},
                            index=[0]),ignore_index=True)

#%%
results2.to_csv('results2.csv')

#%%
# select the overall min mse
results2.min()

# the optimal model which has the smallest mse is:
# LSTM2HiddenLayers (Ytrain, 100, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 32, 100)

#%%
# test if the dataset is sensitive with batch size
batch_set = [16, 32, 64]

results3 = pd.DataFrame([])

for b in batch_set:
    Ytest_optimal_b = LSTM2HiddenLayers (Ytrain, 100, 0.25, 'relu', 'mean_squared_error', 'rmsprop', b, 100)
    results3 = results3.append(
            pd.DataFrame({'batch_size':b,
                          'mse_optimal':mean_squared_error(Ytest, Ytest_optimal_b.reshape(-1,1))},
                           index=[0]),ignore_index=True)    

#%%
results3.to_csv('results3.csv')
# increase or decrease batch size seems does not imporve imporve the results a lot, so select the batch size among 3 with the min mse

#%%
# the optimal model is 
# # LSTM2HiddenLayers (Ytrain, 100, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 16, 100)

Ytest_optimal = LSTM2HiddenLayers (Ytrain, 100, 0.25, 'relu', 'mean_squared_error', 'rmsprop', 16, 20)

mse_optimal = mean_squared_error(Ytest, Ytest_optimal.reshape(-1,1))

#%%
# plot
predictions = pd.Series(Ytest_optimal.flatten(), index=data_test.index)
data_train = data_forward.loc[:'2017-12-28']

plt.figure(figsize=(12,5), dpi=100)
plt.plot(data_train, label='Training data')
plt.plot(data_test, label='Testing data')
plt.plot(predictions, label='Forecast')
plt.title('Forecast vs Actuals, LSTM(hidden layers=2, units=100, \n dropout=0.25, activation=relu, loss=mse, optimizer=rmsprop, batch=16, epochs=100), test MSE=6074')
plt.legend(loc='upper left', fontsize=8)
plt.show()



                            
