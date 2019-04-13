# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:33:00 2019

@author: jc
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from chapter07.nn.conv.resnet import ResNet
from chapter07.callbacks import 