#import the third-party packages
import numpy as np
import cv2

#capture video using the webcam in laptop 
cap = cv2.VideoCapture(0)

#use  opencv's haar frontface cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (True):
	#read the current frame and check 
	ret, frame = cap.read()
	if not ret: break

	#transform the img to gray as color is not necessary for face detection
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)	
	
	for face in faces:
		(x, y, w, h) = face
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	#show the image prosessed
	cv2.imshow("maybe some faces",frame)

	#if 'q' pressed , break the loop
	keyPressed = cv2.waitKey(30) & 0xff
	if keyPressed == ord('q'):
		break

#clean up the cap and close any open windows
cap.release()
cv2.destroyAllWindows()