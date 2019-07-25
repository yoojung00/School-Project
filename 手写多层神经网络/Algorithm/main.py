#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from MLP import MLP
import os
#%%
def load_data():
    with h5py.File('..' + os.sep + 'Input' + os.sep + 'train_128.h5', 'r') as H:
        train_data = np.copy(H['data'])
    with h5py.File('..' + os.sep + 'Input'+ os.sep + 'train_label.h5', 'r') as H:
        label = np.copy(H['label'])
    with h5py.File('..' + os.sep + 'Input' + os.sep + 'test_128.h5', 'r') as H:
        test_data = np.copy(H['data'])
        
    print('train data: ' + str(train_data.shape))
    print('train label: ' + str(label.shape))
    print('test data: ' + str(test_data.shape))
    
    return train_data, label, test_data
#%%
def normalise(x):
    z = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return z
#%%
def accuracy(y, y_pred):
    return np.mean(y == y_pred)

#%%
#load data
train_data, label, test_data = load_data()

#transfer label into one-hot-encoder
label_one_hot = np.zeros((label.size, label.max() + 1))
label_one_hot[np.arange(label.size), label] = 1

#normalise data
train_data = normalise(train_data)
test_data = normalise(test_data)
#%%
#Split data
train_index = 50000
validation_index = 50000
# Training data
input_data = train_data[:train_index, :]
output_data = label_one_hot[:train_index, ]
#Validation data for testing.
vali_data = train_data[validation_index:, :]
vali_label = label[validation_index:]
#%%
# Using different configuration of model for training
# Benchmark model: batch GD, 4 layers, MSE loss, learning rate 0.01
benchmark_nn = MLP([128,60,50,10], [None,'relu','relu','relu'])
benchmark_nn_loss = benchmark_nn.fit(input_data, output_data, learning_rate=0.01, epochs=80, \
                                minibatches_size=0,loss_function='MSE', lb=0, p=1, \
                                batch_norm=False, momentum=False, r=0 )
# Training
train_output = benchmark_nn.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = benchmark_nn.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for benchmark model:', acc_train)
print('Validation accuracy for benchmark model:', acc_test)

#%%
# Model 1: 4 layers, cross-entropy loss, mini-batch training (256).
model1 = MLP([128,80,70,10], [None,'relu','relu','softmax'])
model1_loss = model1.fit(input_data, output_data, learning_rate=0.01, epochs=150, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0, p=1, \
                                batch_norm=False, momentum=False, r=0.9 )
# Training
train_output = model1.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model1.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for model 1:', acc_train)
print('Validation accuracy for model 1:', acc_test)

#%%
# Model 2: 4 layers, tanh activation, cross-entropy loss, mini-batch training (256).
model2 = MLP([128,80,70,10], [None,'tanh','tanh','softmax'])
model2_loss = model2.fit(input_data, output_data, learning_rate=0.01, epochs=150, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0, p=1, \
                                batch_norm=False, momentum=False, r=0.9 )
# Training
train_output = model2.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model2.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for model 2:', acc_train)
print('Validation accuracy for model 2:', acc_test)

#%%
# Model 3: 5 layers, cross-entropy loss, mini-batch training (256), dropout(0.6), weight decay(0.00004).
model3 = MLP([128,80,90,70,10], [None,'relu','relu','relu','softmax'])
model3_loss = model3.fit(input_data, output_data, learning_rate=0.1, epochs=150, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0.00004, p=0.6, \
                                batch_norm=False, momentum=False, r=0.9 )
# Training
train_output = model3.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model3.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for model 3:', acc_train)
print('Validation accuracy for model 3:', acc_test)

#%%
# Model 4: 5 layers, cross-entropy loss, mini-batch training (256), dropout(0.6), weight decay(0.00004), batch normalizetion and momentum update.
model4 = MLP([128,80,90,70,10], [None,'relu','relu','relu','softmax'])
model4_loss = model4.fit(input_data, output_data, learning_rate=0.1, epochs=150, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0.00004, p=0.6, \
                                batch_norm=True, momentum=False, r=0.9 )
# Training
train_output = model4.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model4.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for model 4:', acc_train)
print('Validation accuracy for model 4:', acc_test)
#%%
# Model 5: 5 layers, cross-entropy loss, mini-batch training (256) and momentum update.
model5 = MLP([128,60,50,10], [None,'relu','relu','softmax'])
model5_loss = model5.fit(input_data, output_data, learning_rate=0.1, epochs=80, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0, p=1, \
                                batch_norm=False, momentum=True, r=0.9 )
# Training
train_output = model5.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model5.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy for model 5:', acc_train)
print('Validation accuracy for model 5:', acc_test)
#%%
# Model for testing: 5 layers, cross-entropy loss, mini-batch training (256), dropout(0.6), weight decay(0.00004), batch normalizetion and momentum update.
model_test = MLP([128,80,90,70,10], [None,'relu','relu','relu','softmax'])
model_test_loss = model_test.fit(input_data, output_data, learning_rate=0.1, epochs=150, \
                                minibatches_size=256,loss_function='cross_entropy', lb=0.00004, p=0.6, \
                                batch_norm=True, momentum=True, r=0.9 )
# Training
train_output = model_test.predict(input_data)
acc_train = accuracy(label[:train_index], train_output)
#Testing
vali_output = model_test.predict(vali_data)
acc_test = accuracy(vali_label, vali_output)
print('Training accuracy of model for testing:', acc_train)
print('Validation accuracy of model for testing:', acc_test)
#%%
#Visualization
plt.figure(figsize = [8, 4])
#plt.plot(benchmark_nn_loss, label='benchmark model')
plt.plot(model1_loss, label='model1')
plt.plot(model2_loss, label='model2')
plt.plot(model3_loss, label='model3')
plt.plot(model4_loss, label='model4')
plt.plot(model5_loss, label='model5')
plt.plot(model_test_loss, label='Model for testing')

plt.legend(loc="upper right")
plt.show(block=False)



#%%
#Predict the test data and save to csv file
start_predict_time = time()

predict_label = model_test.predict(test_data)

end_predict_time = time()
print("The testing used: %2f seconds" % (end_predict_time - start_predict_time))

output_data = pd.DataFrame(predict_label)
output_data.to_csv("../Output/Predicted_labels.h5", sep=",", index = False, header=False)
