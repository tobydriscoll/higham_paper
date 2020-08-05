#!/usr/bin/env python
# coding: utf-8

# In[4]:


##Import all necessary packages
import os
import glob
import random
import pandas as pd
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import PIL as image
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[5]:


tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus)


# In[11]:


##Path to dataset
data_dir = '/storage/set_1_20J_test_train/'
train_path = data_dir+'train/classes/'
test_path = data_dir+'test/classes'


# In[12]:


image_shape = (192,192,3)


# In[ ]:





# In[13]:


#Create image generator with all necessary transformations
image_gen = ImageDataGenerator(rotation_range=45, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               horizontal_flip=True, # Allow horizontal flipping
                               vertical_flip=True, # Allow vertical flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                            )


# In[14]:


image_gen.flow_from_directory(train_path)


# In[15]:


image_gen.flow_from_directory(test_path)


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization


# In[17]:


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = Sequential()

    model.add(AveragePooling2D(pool_size = (2,2),strides = 2,input_shape = image_shape)) 

    model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1), padding='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3,3), strides=2, padding='same'))

    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1, 1)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=2, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1)))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())

    # Dropouts help reduce overfitting by randomly turning neurons off during training.
    # Here we say randomly turn off 40% of neurons.
    model.add(Dropout(0.4))

    # Last layer, remember its 5 classes so we use softmax
    model.add(Dense(5))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    


# In[18]:


model.summary()


# In[19]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss',patience=2)


# In[20]:


batch_size = 100


# In[21]:


train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size = (192,192),
                                               batch_size=batch_size)


# In[22]:


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size = (192,192),
                                               batch_size=batch_size)


# In[23]:


train_image_gen.class_indices


# In[24]:


import warnings
warnings.filterwarnings('ignore')


# In[26]:


results = model.fit(train_image_gen,epochs=50,
                              validation_data=test_image_gen,
                              callbacks=[checkpoint])


# In[ ]:




