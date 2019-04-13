# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 09:59:41 2019

@author: jc
"""

import matplotlib
matplotlib.use("Agg")

from dlcv.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from dlcv.nn.conv.minivggnet import MiniVGGNet
from keras.datasets import cifar10
from keras.optimizers import SGD
import argparse
import os



print("[INFO] loading CIFAR10 data...")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float')/255.0
testX = testX.astype('float')/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)



labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# initialize the SGD optimizer, but without any learning rate decay
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# train the network
print("[INFO] training network...")
model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, 
          epochs=100, callbacks=callbacks, verbose=1)












 