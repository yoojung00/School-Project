#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:55:01 2019

@author: heyujun
"""
import numpy as np
import pandas as pd
import tarfile
import os
from PIL import Image, TarIO
import matplotlib.pyplot as plt
import tensorflow as tf

#%%
'''
# Extract file
tar = tarfile.open('train.tar.gz',"r:gz")
tar.extractall()
tar.close()
'''
#%%
# Define a function to get dataset
def get_dataset(width, hight, num_samples, skip_samples=None):
    image = []
    label = []
    
    file_list = pd.read_csv("train.txt", sep = '\t', header=None, names = ['image', 'label'],\
                            skiprows=skip_samples, nrows=num_samples)
    
    file_list["label"]=file_list["label"].apply(lambda x:x.split(","))
    print(file_list)
    
    for pic in os.listdir('train2014/'):
        if pic.endswith('.jpg') and any(file_list['image']==pic):
            Im = Image.open('train2014/'+ pic)
            Im = np.array(Im.resize((width,hight),Image.NEAREST).convert('RGB'))
            image.append(Im)
            label.append(file_list[file_list['image']==pic]['label'].values)
    image = np.array(image).astype(np.float64)
    image = np.array(image) * 1./255.
    return image, label

#%%
def one_hot_label(label):
    
    max_int = 0
    label_int = []
    for i in label:
        labels = i[0].split(',')
        vec = []
        for j in labels:
            vec.append(int(j))
            if int(j) > max_int:
                max_int = int(j)
        label_int.append(vec)

    one_hot = []
    for label in label_int:
        vec = np.zeros(max_int+1)  
        for i in label:
            vec[i] = 1
        one_hot.append(vec)

    one_hot = np.array(one_hot)
    return one_hot

#%%
def build_resnet50(image_batch, n_classes=20, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = tf.keras.applications.ResNet50(include_top=False, weights=weights,
                       input_tensor=None, input_shape=image_batch[1:])
    
    
    # Feature extraction layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # Dense layer
    #dense_layer = tf.keras.layers.Dense(1024, activation="relu")
    
    # Dropout
    #dropout_layer = tf.keras.layers.Dropout(0.5)
    
    # Fully connect layer
    prediction_layer = tf.keras.layers.Dense(n_classes, activation='softmax')
    
    model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True
               
    return model
#%%
def norm(x):
    return (x-x.min())
#%%
# Get validation data
val_image, v_label = get_dataset(224, 224, 5)
#%%
# One-hot transfer
val_label = one_hot_label(v_label)
#%%  

learning_rate = 0.0001
model = build_resnet50(val_image.shape, load_pretrained=True)
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),\
              loss='categorical_crossentropy',\
              metrics=['accuracy'])
#%%
for i in range(0, 1):
    train_image, t_label = get_dataset(224, 224, 600, i)
    train_label = one_hot_label(t_label)
    #train_std, val_std = train_val_stand(train_image, val_image)
    history = model.fit(train_image, train_label, batch_size=32, epochs=1)
    
#%%
loss =np.array(loss).reshape(-1,1)
val_loss = np.array(val_loss).reshape(-1,1)

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#%%
# Validation testing
true_label = []
for i in range(val_label.shape[0]):
    v = np.where(val_label[i]==1)
    true_label.append(v)  
#%%
label_pro = model.predict(val_image)
label_pred = np.argmax(label_pro, axis = 1)
#%%
count = 0
for i in range(label_pred.shape[0]):
    if label_pred[i] in true_label[i][0]:
        count += 1
print(count/label_pred.shape[0])

#%%
# Save model
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
#%%
# Load model
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

