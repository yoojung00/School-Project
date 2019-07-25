#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 21:30:38 2019

@author: heyujun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick2_ohlc
import os
from PIL import Image
import talib

from keras import optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, add
from keras.layers import Dropout, Flatten, LSTM, ReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
#%%
raw_data = pd.read_excel('audusd.xlsx', index_col =0, names = ['date', 'open', 'high', 'low', 'close'],\
                         dtype = {'date':'str'}, usecols = [2, 3, 4, 5, 6])
data = raw_data.loc['2000-01-01':]
train_number = len(data.loc[:'2018-01-01'])

#%%
# Add moving average indicators
data['sma5'] = talib.SMA(data['close'], timeperiod=5)
data['sma10'] = talib.SMA(data['close'], timeperiod=10)
data['sma15'] = talib.SMA(data['close'], timeperiod=15)
data['sma20'] = talib.SMA(data['close'], timeperiod=20)
data['sma25'] = talib.SMA(data['close'], timeperiod=25)
data = data.dropna()
data = data.round({'sma5': 3, 'sma10': 3, 'sma15': 3, 'sma20': 3, 'sma25': 3})
#%%
data.loc[(data['sma5']==data['sma10']), 'highly_predict'] = 1
data.loc[(data['sma5']==data['sma15']), 'highly_predict'] = 1
data.loc[(data['sma5']==data['sma20']), 'highly_predict'] = 1
data.loc[(data['sma5']==data['sma25']), 'highly_predict'] = 1
data.loc[(data['sma10']==data['sma15']), 'highly_predict'] = 1
data.loc[(data['sma10']==data['sma20']), 'highly_predict'] = 1
data.loc[(data['sma10']==data['sma25']), 'highly_predict'] = 1
data.loc[(data['sma15']==data['sma20']), 'highly_predict'] = 1
data.loc[(data['sma15']==data['sma25']), 'highly_predict'] = 1
data.loc[(data['sma20']==data['sma25']), 'highly_predict'] = 1
#%%
# If you have not created picture,please create firstly.
# Create caddlestick picture
def caddlestick_data(data, days_for_train, days_for_label, p, highly_predict=False):
    """
    input:
        data: contain open, high, low, close.
        days_for_train: window size for training.
        days_for_label: window size for label.
        p: the up percentage for seperating label 1 and 0.
        highly_predict: True is to only create highly predicted images.
    return: caddelstick image dataset.
    """
    if highly_predict:
        highpred_list = data[days_for_train:-days_for_label].loc[data['highly_predict']==1].index
        for i,d in enumerate(highpred_list):
            c = data.ix[:d][-days_for_train:]
            plt.style.use('dark_background')
            fig = plt.figure()
            ax = plt.axes()
        
            ax.plot(c['sma5'].values, linewidth=2.2)
            ax.plot(c['sma10'].values, linewidth=2.2)
            ax.plot(c['sma15'].values, linewidth=2.2)
            ax.plot(c['sma20'].values, linewidth=2.2)
        
        
            candlestick2_ohlc(ax, c['open'], c['high'], c['low'], c['close'] , width = 0.8, colorup = 'g', colordown = 'r')
            plt.axis('off')
            
            l = data.ix[d:][:days_for_label+1]
            change = ((l['close'].shift(-1)-l['close'])/l['close']).dropna()   
    
            if any(e > p for e in change):
                pngfile = '{}-{}_{}.png'.format('h', '%05d'%i, 1)
                fig.savefig(pngfile, bbox_inches='tight',pad_inches = 0)
            else:
                pngfile = '{}-{}_{}.png'.format('h', '%05d'%i, 0)
                fig.savefig(pngfile, bbox_inches='tight',pad_inches = 0)
            plt.close(fig)
    
    else:
        for i in range(0, len(data) - days_for_train - days_for_label - 1):
            c = data.ix[i:i + int(days_for_train), :]
            plt.style.use('dark_background')
            fig = plt.figure()
            ax = plt.axes()
        
            ax.plot(c['sma5'].values, linewidth=1.8)
            ax.plot(c['sma10'].values, linewidth=1.8)
            ax.plot(c['sma15'].values, linewidth=1.8)
            ax.plot(c['sma20'].values, linewidth=1.8)
        
        
            candlestick2_ohlc(ax, c['open'], c['high'], c['low'], c['close'] , width = 0.5, colorup = 'g', colordown = 'r')
            plt.axis('off')
        
            change = []
            for j in range(1, days_for_label+1):
                change.append((data['close'][i + int(days_for_train) + j] - \
                           data['close'][i + int(days_for_train)]) /data['close'][i + int(days_for_train)])
    
            if any(e > p for e in change):
                pngfile = '{}_{}.png'.format('%05d'%i, 1)
                fig.savefig(pngfile, bbox_inches='tight',pad_inches = 0)
            else:
                pngfile = '{}_{}.png'.format('%05d'%i, 0)
                fig.savefig(pngfile, bbox_inches='tight',pad_inches = 0)
            plt.close(fig)
    print("Finish Creating Picture.")
    
#%%
#Save picture
days_for_train = 20
days_for_label = 5
p = 0.01

#create image data
caddlestick_data(data, days_for_train, days_for_label, p)

#create highly-predicted image data
#caddlestick_data(data, days_for_train, days_for_label, p, highly_predict=True)

#%%
def get_dataset(width, hight, highly_predict=False):
    image = []
    label = []
    index = []
    
    if highly_predict:
        for pic in os.listdir('.'):
            if pic.endswith('.png') and pic.split("-")[0]=='h':
                Im = Image.open(pic)
                Im = np.array(Im.resize((width,hight),Image.NEAREST).convert('RGB'))
                image.append(Im)
                label.append(int(pic.split("-")[1].split("_")[1].split(".")[0]))
                index.append(int(pic.split("-")[1].split("_")[0]))    
        
    else:
        for pic in os.listdir('.'):
            if pic.endswith('.png') and pic.split("-")[0]!='h':
                Im = Image.open(pic)
                Im = np.array(Im.resize((width,hight),Image.NEAREST).convert('RGB'))
                image.append(Im)
                label.append(int(pic.split("_")[1].split(".")[0]))
                index.append(int(pic.split("_")[0]))
                
    label = np.array(label)
    image = np.array(image)
    index = np.array(index)
    #dataset = pd.DataFrame({'index': index,'image': image,'label': label})
    return index, image, label

#%%
#load data
index, image, label = get_dataset(128, 128, highly_predict=False)
    
#load highly_predicted data
#index_h, image_h, label_h = get_dataset(128, 128, highly_predict=True)
#%%
#Sort thhe image
index_sort = np.argsort(index)
image = image[index_sort]
label = label[index_sort]

#index_sort_h = np.argsort(index_h)
#image_h = image_h[index_sort_h]
#label_h = label_h[index_sort_h]
#%%
label_onehot = np.zeros((label .size, label .max() + 1))
label_onehot[np.arange(label .size), label] = 1

#label_onehot_h = np.zeros((label_h .size, label_h .max() + 1))
#label_onehot_h[np.arange(label_h .size), label_h] = 1
#%%
#normalize the data
def norm(x):
    return x/255.

image_norm = norm(image)
#image_norm_h = norm(image_h)
#%%    
#seperate picture to train and test
x_train = image_norm[0:train_number]
y_train = label_onehot[0:train_number]

x_test = image_norm[train_number:]
y_test = label_onehot[train_number:]

#x_train_h = image_h[0:1500]
#y_train_h = label_onehot_h[0:1500]

#x_test_h = image_h[1500:]
#y_test_h = label_onehot_h[1500:]
#%%
def simple_CNN(input_shape):
    X_input = Input(input_shape)
    
    X = Conv2D(32,(3,3), strides = (1,1))(X_input)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64,(3,3), strides = (1,1))(X_input)
    X = MaxPooling2D((2,2))(X)
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Dense(2, activation = 'softmax')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'baseline')
    
    return model
#%%
def deep_CNN(SHAPE, seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # Step 1
    x = Conv2D(32, (3, 3), init='glorot_uniform',border_mode='same')(input_layer)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 1
    x = Conv2D(48, (3, 3), init='glorot_uniform', border_mode='same')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Step 1
    x = Conv2D(64, (3, 3), init='glorot_uniform', border_mode='same')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 1
    x = Conv2D(96, (3, 3), init='glorot_uniform', border_mode='same')(x)
    # Step 2 - Pooling
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    # Step 3 - Flattening
    x = Flatten()(x)

    # Step 4 - Full connection

    x = Dense(output_dim=256)(x)
    #x = BatchNormalization()(x)
    #x = ReLU()(x)
    # Dropout
    x = Dropout(0.5)(x)

    x = Dense(output_dim=2, activation='softmax')(x)

    model = Model(input_layer, x)

    return model

#%%
def VGG16(SHAPE, seed=None):
    # We can't use ResNet50 directly, as it might cause a negative dimension
    # error.
    if seed:
        np.random.seed(seed)

    input_layer = Input(shape=SHAPE)

    # block 1
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv1')(input_layer)
    x = Conv2D(64, (3, 3),
               activation='relu',
               padding='same',
               name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # block 2
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv1')(x)
    x = Conv2D(128, (3, 3),
               activation='relu',
               padding='same',
               name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # block 3
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv1')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv2')(x)
    x = Conv2D(256, (3, 3),
               activation='relu',
               padding='same',
               name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv1')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv2')(x)
    x = Conv2D(512, (3, 3),
               activation='relu',
               padding='same',
               name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1, activation='softmax', name='predictions')(x)

    model = Model(input_layer, x)

    return model      
        
#%%
#train
model = deep_CNN(x_train.shape[1:])
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])  
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 100, batch_size=16)
#%%

# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

plt.figure()
plt.style.use("seaborn-whitegrid")
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
#%%
loss, test_accuracy = model.evaluate(x_test, y_test, batch_size = 1, verbose=1)
#%%
#test
y_pred = model.predict(x_test, batch_size = 1, verbose=1)

y_pred[y_pred>0.5]=1
y_pred[y_pred<0.5]=0

from sklearn.metrics import classification_report
target_names = ['0', '1']
print(classification_report(y_test, y_pred, target_names=target_names))
#%%
#test accuracy:0.6162361623616236
#test accuracy for highly predict:0.5131086142322098