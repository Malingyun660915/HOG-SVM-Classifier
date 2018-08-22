# Import necessary packages
import cv2
from sklearn.externals import joblib
from threading import Thread
import timeit
import os
import numpy as np

currFileDir = os.path.dirname(__file__)  + '/' # For some reason relative paths are not working

# Load the classifier
filename = 'faceData/svmModel.pkl'
svmClf = joblib.load(currFileDir + filename)

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

# A function that takes in the co-ordinates of the face, compute its HOG Features, predict the person and print its name.
def faceRecFun(faces):
	xf=faces[0]
	yf=faces[1]
	wf=faces[2]
	hf=faces[3]
	img2=gray[yf:yf+hf,xf:xf+wf]
	img2=cv2.resize(img2,(200,200))
	hist = hog.compute(img2,winStride,padding,locations)
	hist = np.reshape(hist,(1,1764))
	predName = svmClf.predict(hist)[0]
	print(predName)
	cv2.rectangle(img1,(xf,yf),(xf+wf,yf+hf),(225,0,0),2)
	cv2.putText(img1,str(predName), (xf,yf+hf+25),font,1,(255,255,255),1,cv2.LINE_AA)
	cv2.imshow("image",img1)
	
	
# Open the camera  
cam=cv2.VideoCapture(0)

# Load HAAR Cascade
getFace=cv2.CascadeClassifier(currFileDir+'faceData/haarcascade_frontalface_default.xml')

# Set font for printing names
font = cv2.FONT_HERSHEY_SIMPLEX

# Open an image window. Detect all the faces, put a rectangle on them and lable them till the user presses q.
while(True):
	start = timeit.default_timer()
	ret, img1=cam.read()
	gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	faces=getFace.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5)
	if np.shape(faces)[0]>1:
		for face in faces:
			t = Thread(target=faceRecFun, args = (face,))
			t.start()
	elif(faces!=()):        
		faceRecFun(faces[0])

	# Stop timer and print time difference
	stop = timeit.default_timer()
	print(stop - start)

	# Break the loop if the user pressed q
	if cv2.waitKey(1) & 0xff==ord('q'):
		break

# Clean Up
cam.release()
cv2.destroyAllWindows()
