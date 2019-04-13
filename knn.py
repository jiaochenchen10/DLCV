# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:29:58 2018

@author: jc
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input datasets")
ap.add_argument("-k","--neighbors",type=int,default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j","--jobs",type=int,default=-1,
                help="# of jobs for knn distance(-1 uses all avaiable cores)")
args=vars(ap.parse_args())
 
# Grab the list of each image's absolute path.
# for example: imagePaths[:2]=['F:\\pyimagesearch\\datasets\\animals\\cats\\cats_00001.jpg', 'F:\\pyimagesearch\\datasets\\animals\\cats\\cats_00002.jpg']
print("[INFO] loading image...")
imagePaths=list(paths.list_images(args["dataset"]))
#print(imagePaths[:3])

sp=SimplePreprocessor(32,32)
sdl=SimpleDatasetLoader(preprocessors=[sp])
(data,labels)=sdl.load(imagePaths,verbose=500)
data=data.reshape((data.shape[0],32*32*3))

print("[INFO] feature matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))

# Encode the labels as integers.
le=LabelEncoder()
labels=le.fit_transform(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

model=KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))


