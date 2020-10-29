# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:27:01 2020

@author: ADITYA SINGH
"""


from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K

IMAGE_SIZE = [128, 128]

input_shape=IMAGE_SIZE + [3]
folders = glob('Architectural_Heritage_Elements_Dataset_128(creative_commons)/*')

train_path = 'Architectural_Heritage_Elements_Dataset_128(creative_commons)'

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(len(folders)))

model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',

          optimizer='rmsprop',

          metrics=['accuracy'])

model.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

r = model.fit(
  training_set,
  epochs=15,
  steps_per_epoch=len(training_set))