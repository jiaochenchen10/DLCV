# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:51:15 2019

@author: jc
"""

import numpy as np
import cv2
import os

class DatasetLoader():
    def __init__(self, preprocessors=None):
        """
        parameter：
            preprocessors:图片处理器对象列表，为None时不做处理
        """
        self.preprocessors = preprocessors
        
        # if preprocessors are None, initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors = []
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []
        
        # loop over the input image paths
        for (i, imagepath) in enumerate(imagePaths):
            
            # assuming path has the format:
            # path/dataset_name/class_name/image.jpg
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.sep)[-2]
            
            # if necessary, preprocess the input image
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)
            
            # show the number of loaded images
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print(f"[INFO] processed {i+1}/{len(imagePaths)}...")
        
        return np.array(data), np.array(labels)
                
                
            
        
            
        