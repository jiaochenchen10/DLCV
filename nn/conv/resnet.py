# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:45:13 2019

@author: jc
"""

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

class ResNet:
    @staticmethod
    def residual_module(data,k,stride,chanDim,red=False,
                        reg=0.0001,bnEps=2e-5,bnMom=0.9):
        """
        data: 
            input to the residual network.
        k: 
            the number of filter that will be 
            learned by final CONV in the bottleneck.
        stride: 
            reduce the spatial dimensions of our volume without 
            resorting to max pooling.
        chanDim: 
            define the axis which will perform batch normalization.
        red: 
            (reduce=True or False) 
            control whether we are reducing spatial dimensions.
        reg: 
            the regularization strength to all CONV layers.
        bnEps: 
            avoid division by zero when normalize inputs.
        bnMom: 
            the momentum for moving average.
        
        """
        # the shortcut branch of the Resnet module 
        # should be initialize as input(identity) data
        shortcut=data
        
        # the first block of the resnet module are the 1*1 CONVs
        bn1=BatchNormalization(axis=chanDim,epsilon=bnEps,
                               momentum=bnMom)(data)
        act1=Activation("relu")(bn1)
        conv1=Conv2D(int(k*0.25),(1,1),use_bias=False,
                     kernel_regularizer=l2(reg))(act1)
        bn2=BatchNormalization(axis=chanDim,
                               epsilon=bnEps,momentum=bnMom)(conv1)
        act2=Activation("relu")(bn2)
        conv2=Conv2D(int(k*0.25),(3,3),stride=stride,padding="same",
                     use_bias=False,kernel_regularizer=l2(reg))(act2)     
        
        # the third block of the Resnet module is another set of 1*1 CONVs
        bn3=BatchNormalization(axis=chanDim,
                               epsilon=bnEps,momentum=bnMom)(conv2)
        act3=Activation("relu")(bn3)
        conv3=Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)
        
        # if we are to reduce the spatial size, 
        # apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(k,(1,1),stride=stride,
                            use_bias=False,kernel_regularizer=l2(reg))(act1)
            x = add([conv3,shortcut])
            return x
        
        @staticmethod
        def build(width,height,depth,classes,stages,
                  filters,reg=0.0001,bnEps=2e-5,bnMom=0.9,dataset="cifar"):
            """
            stages( type: list ): [3,4,6] 
            filters( type: list): [64,128,256,512]
            
            """
            inputShape = (height, width, depth)
            chanDim = -1
            if K.image_data_format() == "channels_first":
                inputShape = (depth,height,width)
                chanDim = 1
                
            inputs = Input(shape=inputShape)   
            x=BatchNormalization(axis=chanDim,epsilon=bnEps,
                                 momentum=bnMom)(inputs)
                
            # check if wo are utilizing the CIFAR dataset
            if dataset == "cifar" :
                # apply a single CONV layer
                x = Conv2D(filters[0],(3,3),use_bias=False,
                           padding="same",kernel_regularizer=l2(reg))(x)
                
            # loop over the number of stages
            for i in range(0,len(stages)):
                # initialize the stride, then apply a residual module to 
                # reduce the spatial size of the input volumn
                stride = (1,1) if i == 0 else (2,2)
                x = ResNet.residual_module(x,filters[i+1],stride,
                           chanDim,red=True,bnEps=bnEps,bnMom=bnMom)
                
                # loop over the number of layers in the stage
                for j in range(0,stages[i]-1):
                    # apply a ResNet module
                    x = ResNet.residual_module(x,filters[i+1],
                           (1,1),chanDim,bnEps=bnEps,bnMom=bnMom)
                
            # apply BN=> ACT => POOL
            x=BatchNormalization(axis=chanDim,
                                 epsilon=bnEps,momentum=bnMom)(x)
            x=Activation("relu")(x)
            x=AveragePooling2D((8,8))(x)
            x=Flatten()(x)
            x=Dense(classes,kernel_regularizer=l2(reg))(x)
            x=Activation("softmax")(x)
            model=Model(inputs,x,name="resnet")
            print(model.summary())
            return model
        from keras.applications import inception_v3
        
