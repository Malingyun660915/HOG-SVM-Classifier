# Import necessary packages
import cv2
import os

currFileDir = os.path.dirname(__file__)  + '/' # For some reason relative paths are not working

# Open the camera
cam = cv2.VideoCapture(0)

# Load HAAR Casacde 
detector=cv2.CascadeClassifier(currFileDir+'faceData/haarcascade_frontalface_default.xml')

# Set sample name
i=1

# Start capturing images. Convert the image to gray scale and mark all the faces in a rectangle. Quit when user presses q.
while(True):
	ret, img = cam.read()
	cv2.imshow("Image", img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5)

	# Save the image to the dataset. Mark a rectangle around the face.
	for (x,y,w,h) in faces:
		img=gray[y:y+h,x:x+w]
		cv2.imwrite(currFileDir+'faceData/u1/'+str(i)+'.pgm', img)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	# Set name for next face image.
	i=i+1

	# Break the loop if the user pressed q
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Close all the active image windows.
cam.release()
cv2.destroyAllWindows()
