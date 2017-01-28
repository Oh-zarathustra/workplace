#import the third-party packages
import numpy as np
import cv2
import time
#capture video using the webcam in laptop 
#after background is ready
# time.sleep(5)
cap = cv2.VideoCapture(0)

# creat a background subtractor using the MOG algorithm
# all the parameters are set to default excpet we don't 
# detect Shadows
bgs = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
while (1):
	#read the current frame and check 
	ret, frame = cap.read()
	if not ret: break

	#smooth the img
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)

	#get the frontground mask using MOG
	fg = bgs.apply(blur)

	cnts = cv2.findContours(fg.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[1]

	# loop over the contours
	for cnt in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(cnt) < 1000:
			continue

		# compute the bounding ellipse for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(cnt)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	#show the image prosessed
	cv2.imshow("prossessing",frame)

	#if 'q' pressed , break the loop
	keyPressed = cv2.waitKey(30) & 0xff
	if keyPressed == ord('q'):
		break

#clean up the cap and close any open windows
cap.release()
cv2.destroyAllWindows()