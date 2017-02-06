import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

	# Take each frame
	_, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#b1 = cv2.GaussianBlur(gray, (11, 11), 0)
	b1 = cv2.bilateralFilter(gray, 7,41,41)
	b2 = cv2.bilateralFilter(frame, 7,41,41)
	b2 = cv2.cvtColor(b2, cv2.COLOR_BGR2GRAY)
	edged1 = cv2.Canny(b1, 30, 150)
	edged2 = cv2.Canny(b2, 30, 150)

	#cv2.imshow('frame',frame)
	cv2.imshow('b1',b1)
	cv2.imshow('b2',b2)
	cv2.imshow('edged1',edged1)
	cv2.imshow('edged2',edged2)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()