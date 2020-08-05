#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from tensorflow import keras
import sys
import matplotlib as plt
from matplotlib import image
from matplotlib import pyplot
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets
import tensorflow.keras.backend as K
from sklearn.metrics import classification_report,confusion_matrix


# In[2]:


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus)


# In[3]:


def get_data():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images,train_labels,test_images,test_labels)


# In[4]:


mods = ImageDataGenerator(featurewise_center=True,              #mean centered at 0            
                          featurewise_std_normalization=False,   #std deviation normalization
                          zca_whitening = False,
                          rotation_range=20,
                          width_shift_range=0.2,
                          height_shift_range=0.2,
                          horizontal_flip=True)


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense,Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization


# In[6]:


def scheduler(epoch):
    if epoch < 30:
        return .05
    elif epoch < 40:
        return .005
    else:
        return .0005
custom_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

#create callback class to display learing rate at each epoch
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        lr = float(K.get_value(self.model.optimizer.lr))
        print(" Learning rate:", lr)


# In[9]:


def new_model():
    model = Sequential()

    #block_1
    #layer_1
    model.add(Conv2D(filters = 32, kernel_size = (5,5), strides = (1,1), input_shape= (32,32,3), padding = 'same'))
    model.add(BatchNormalization())
    #layer_2
    model.add(MaxPooling2D(pool_size=(3,3),strides = 2,padding = 'same'))
    #layer_3
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #block_2
    #layer_4
    model.add(Conv2D(filters =32, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    #layer_5
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #layer_6
    model.add(AveragePooling2D(pool_size = (3,3),strides = 2,padding = 'same'))

    #block_3
    #layer_7
    model.add(Conv2D(filters =64, kernel_size = (5,5), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    #layer_8
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    #layer_9
    model.add(AveragePooling2D(pool_size = (3,3),strides = 2,padding = 'same'))

    #block_4
    #layer_10
    model.add(Conv2D(filters =64, kernel_size = (4,4), strides = (1,1)))
    model.add(BatchNormalization())
    #layer_11
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    #block_5
    #layer_12
    model.add(Conv2D(filters =10, kernel_size = (1,1), strides = (1,1)))
    model.add(BatchNormalization())

    #layer_13
    model.add(Flatten())
    #model.add(Dropout(.40))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    optimizer = keras.optimizers.SGD(momentum = 0.09,nesterov = False)

    model.compile(optimizer= optimizer,
                  loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


# In[10]:


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    
    model = new_model()
    #optimizer = keras.optimizers.SGD(momentum = 0.09,nesterov = False)
    #model.compile(optimizer= optimizer,
     #         loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #          metrics=['accuracy'])
    model.summary()


# In[ ]:


with strategy.scope():
    (train_images,train_labels,test_images,test_labels) = get_data()
    mods.fit(train_images)
    mods.fit(test_images)
    callbacks = myCallback() 


# In[14]:


model.fit(mods.flow(train_images,train_labels,batch_size = 100),
          epochs = 4,
          validation_data =(mods.flow(test_images,test_labels,batch_size = 100)))


# In[ ]:




