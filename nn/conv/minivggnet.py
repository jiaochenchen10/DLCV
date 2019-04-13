# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 10:44:10 2018

@author: jc
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class MiniVGGNet():
    @staticmethod
    def build(width,height,depth,classes):
        model=Sequential()
        inputShape=(height,width,depth)
        # The index of channel dimension
        chanDim=-1
        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)
            chanDim=1
        
        # First conv => relu => conv=> relu =>pool => dropput
        model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # Second conv=>relu=>conv=>relu=>pool=>dropout
        model.add(Conv2D(64,(3,3),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        print(model.summary())
        return model        
        
        
        
        
        