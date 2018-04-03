#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 18:19:33 2018

@author: richard
"""

#import
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# first part - buidling the CNN
classifier = Sequential()

classifier.add(Convolution2D(32,3,3, input_shape= (64,64,3), activation= 'relu')) # have to input the Image, no upper layer  #have to input_shape at first time.

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation= 'relu')) # no need for the input of image, so no setting. the up layer is another convolutional layer, not image 
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

# full connection 

classifier.add(Dense(128,activation='relu'))
#classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

classifier.compile(optimizer='adam', loss= 'binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

# get from: https://keras.io/preprocessing/image/            (Keras doucument)
train_datagen = ImageDataGenerator( #used for preprocess the images in test set.      # d
        rescale=1./255,
        shear_range=0.2,            # rescale the image 
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)  #used for preprocess the images in test set.  # rescale the test imange, twitch, tilte, to prevent the overfitting

single_datagen = ImageDataGenerator(rescale=1./255) 

training_set = train_datagen.flow_from_directory( # train set, extract data from train_datagen<ImageGenerator>
        'dataset/training_set',
        target_size=(64, 64), # subjected to the input_shape in Convolution2D
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory( # test set, extract fromd imageGenerator
        'dataset/test_set',  # tune it to single_set
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

single_set = single_datagen.flow_from_directory( # test set, extract fromd imageGenerator
        'dataset/single_set',  # tune it to single_set
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(     # fit and testing the performance
        training_set,
        steps_per_epoch= 8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


# make single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_set/cat.5001.JPG', target_size = (64,64)) #load
test_image = image.img_to_array(test_image) # 3 dimension
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
tra
if result[0][0]==1:
    prediction = 'dog'
esle :
    prediction = 'cat'












