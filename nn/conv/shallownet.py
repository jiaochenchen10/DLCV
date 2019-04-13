# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:20:29 2018

@author: jc
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class ShallowNet():
    @staticmethod
    def build(width,height,depth,classes):
        # Initialize the model along with the         
        # input shape to be "channels_last".
        model=Sequential()
        inputShape=(height,width,depth)
        # If we are using channels_first, update the input shape.
        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)
        # Define the first and the only conv=>relu layers.
        model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        print(model.summary())
        return model

        