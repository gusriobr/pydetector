from __future__ import print_function
import argparse
import cv2

def load_image_matplot(image_path):
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
image = mpimg.imread(image_path)
print("width: {} pixels".format(image.shape[1]))
print("height: {} pixels".format(image.shape[0]))
print("channels: {}".format(image.shape[2]))
plt.imshow(image)
plt.show()

def load_image_opencv(image_path):
	image = cv2.imread("newimage.jpg")
	print("width: {} pixels".format(image.shape[1]))
	print("height: {} pixels".format(image.shape[0]))
	print("channels: {}".format(image.shape[2]))

	cv2.imshow("Image", image)
	cv2.waitKey(0)
	

	
def draw_quiz_image()
import numpy as np
import cv2

red = (0,0,255)
green = (0,255,0)
side = 20
center = 150
canvas = np.zeros((300,300,3),dtype="uint8")

for x in range(0,300,side*2):
	for y in range(0,300,side):
		if (y/side) % 2 ==0:
			cv2.rectangle(canvas,(x,y),(x+side,y+side),red,-1)			
		else:
			cv2.rectangle(canvas,(x+side,y),(x+(side*2),y+side),red,-1)

cv2.circle(canvas,centro,50,green,-1)
			
cv2.imshow("img",canvas); cv2.waitKey(0)

def draw_rect(canvas, side, point, color):
	x = point[0]
	y = point[1]
	if (y/side) % 2 ==0:
		cv2.rectangle(canvas,(x,y),(x+side,y+side),color,-1)			
	else:
		cv2.rectangle(canvas,(x+side,y),(x+(side*2),y+side),color,-1)
	
lst = [(x,y) for x in range(0,300,side*2) for y in range(0,300,side)]
