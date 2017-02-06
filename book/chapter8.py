import numpy as np
import cv2
from matplotlib import pyplot as plt
 
 
def edge_detect(image):
	laplacian = cv2.Laplacian(image,cv2.CV_64F)
	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
	scharrx = cv2.Scharr(image,cv2.CV_64F,1,0)
	sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
	scharry = cv2.Scharr(image,cv2.CV_64F,0,1)
	images = np.hstack([laplacian,sobelx,scharrx,sobely,scharry])
	return images
 

def sobel(img):
#	sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
	#sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
	sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
	sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)	
	sobelx = np.uint8(np.absolute(sobelx))
	sobely = np.uint8(np.absolute(sobely))	
	sc0 = sobelx + sobely
	sc1 = cv2.bitwise_or(sobelx, sobely) 
	sc2 = cv2.bitwise_and(sobelx, sobely) 
	sc3 = cv2.bitwise_xor(sobelx, sobely) 
	sc4 = cv2.bitwise_not(sobelx, sobely) 

	plt.subplot(2,3,1),plt.imshow(sobelx)
	plt.title('Solbel X'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,3,2),plt.imshow(sobely,cmap = 'gray')
	plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,3,3),plt.imshow(sc0,cmap = 'gray')
	plt.title('Merge add'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,3,4),plt.imshow(sc1,cmap = 'gray')
	plt.title('Merge or'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,3,5),plt.imshow(sc2,cmap = 'gray')
	plt.title('Merge and'), plt.xticks([]), plt.yticks([])
	plt.subplot(2,3,6),plt.imshow(sc3,cmap = 'gray')
	plt.title('Merge xor'), plt.xticks([]), plt.yticks([])
	plt.show()
	
	images = np.hstack([sobelx,sobely,sc0,sc1,sc2,sc3,sc4])

	return images
 
 
def mycanny1(image):
	pos = 1
	for i in range(0,100,20):
		canny = cv2.Canny(image, 80, 180+i)
		plt.subplot(2,3,pos),plt.imshow(canny,cmap = 'gray')
		plt.title('Canny 80-'+str(180+i)), plt.xticks([]), plt.yticks([])
		pos=pos+1
	plt.show()
	return


def mycanny2(image):
	pos = 1
	for i in range(0,100,20):
		canny = cv2.Canny(image, 80+i, 200)
		plt.subplot(2,3,pos),plt.imshow(canny,cmap = 'gray')
		plt.title('Canny '+str(80+i)+'-200'), plt.xticks([]), plt.yticks([])
		pos=pos+1
	
	plt.show()
	return
		
 
 
def imshow(text, image):
	cv2.imshow(text, image)
	cv2.waitKey(0)

image = cv2.imread("images/coins.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


lap = cv2.Laplacian(image, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
lap2 = cv2.Laplacian(image, cv2.CV_8U)

images = np.hstack([lap,lap2])

sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

images = np.hstack([laplacian,sobelx,sobely])

images = np.hstack([laplacian,sobelx,sobely])
imshow("Laplacian, Sobelx, Sobely",images)


images = np.hstack([laplacian,sobelx,scharrx,sobely,scharry])
imshow("Laplacian, Sobelx, Scharrx, Sobely, Scharry",images)
