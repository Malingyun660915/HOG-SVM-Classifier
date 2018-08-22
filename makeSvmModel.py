# Import the necessary packages
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.svm import SVC
from sklearn.externals import joblib

currFileDir = os.path.dirname(__file__)  + '/' # For some reason relative paths are not working

# Set parameters for extracting HOG Features 
winSize = (64,64)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64

# Parameters used while computing HOG Features
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

# Prepare HOG Feature Extrator
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

# Prepare Neural Network for classification
svmClf=SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

# Set variables for labels, image data and image id
label=[]
data=[]
k=0

# Load the list of users already trained (edited manually)
users=np.array(pd.read_csv(currFileDir+'faceData/users.csv'))

# Read images one by one and compute its HOG Features. Prepare data and label sets. Train the Neural Net.
for i in range(1,6): # Total 5 users in the dataset 
	path='faceData/u%d/'%i
	imagesPaths=[currFileDir+path+f for f in os.listdir(currFileDir+path)]
	for imagePath in imagesPaths:
		# Double check on the images        
		img=cv2.imread(imagePath,0)
		im=cv2.resize(img,(200,200))
		hist = hog.compute(im,winStride,padding,locations)
		data.append(hist.ravel()) # Save data as an array
		label.append(users[i-1]) # First user on 0th index

svmClf.fit(data,label)

# Save the newly generated classifier as a pickel file
filename = 'faceData/svmModel.pkl'
joblib.dump(svmClf, currFileDir+filename)
