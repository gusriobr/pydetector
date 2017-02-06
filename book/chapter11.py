import numpy as np
import cv2
from matplotlib import pyplot as plt
import mahotas

def imshow(text, image):
	cv2.imshow(text, image)
	cv2.waitKey(0)

	
def thresh(image):
	T = mahotas.thresholding.otsu(lap2)
	th = image.copy()
	th[th>T] = 255
	th[th<=T] = 0
	return th
	
image = cv2.imread("coins.png")
#blurred = cv2.GaussianBlur(gray, (11, 11), 0)
blurred = cv2.bilateralFilter(image,9,75,75)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
im = gray
edged = cv2.Canny(im, 30, 150)
(_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
coins = image.copy()
cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
imshow("coins",coins)