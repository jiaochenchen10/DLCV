# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:24:54 2018

@author: jc
"""

import matplotlib
# Figures can be saved in the background
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
#import argparse

#ap=argparse.ArgumentParser()
#ap.add_argument('-o','--output',required=True,help="path to output loss/accuracy plot")
#args=vars(ap.parse_args())

# Load the data, the scale it into the range [0,1]
print("[INFO] loading CIFAR-10 data...")
(trainX,trainY),(testX,testY) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

print("[INFO] compiling model...")
opt=SGD(lr=0.01,decay=0.01/40,momentum=0.9,nesterov=True)
model=MiniVGGNet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

# Train the nerwork
print("[INFO] training network...")
H=model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=64,epochs=40,verbose=1)

# Evaluate the model
print("[INFO] evaluating the model...")
predictions=model.predict(testX,batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

#Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,40),H.history['loss'],label="train_loss")
plt.plot(np.arange(0,40),H.history['val_loss'],label="val_loss")
plt.plot(np.arange(0,40),H.history['acc'],label='train_acc')
plt.plot(np.arange(0,40),H.history['val_acc'],label='val_acc')
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuarcy")
plt.legend()
plt.savefig('F:/pyimagesearch/chapter07/jkl.png')








