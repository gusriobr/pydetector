
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur



thresh1 = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

thresh2 = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
images = np.vstack([thresh1, thresh2])

cv2.imshow("Gaussian Thresh", thresh);cv2.waitKey(0)

image = cv2.imread("newimage.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def imshow(image):
	cv2.imshow("Gaussian Thresh", image)
	cv2.waitKey(0)

def th_image(img):
	blurred = cv2.GaussianBlur(img, (5, 5), 0)
	T = mahotas.thresholding.rc(blurred)
	thresh = img.copy()
	thresh[thresh > T] = 255
	thresh[thresh < 255] = 0
	#thresh = cv2.bitwise_not(thresh)
	return thresh

th1 = th_image(image)
th2 = th_image(th1)
th3 = th_image(th2)

images = np.hstack([th1,th2,th3])
imshow(images)

26 print("Riddler-Calvard: {}".format(T))
27 thresh = image.copy()
28 thresh[thresh > T] = 255
29 thresh[thresh < 255] = 0
30 thresh = cv2.bitwise_not(thresh)
31 cv2.imshow("Riddler-Calvard", thresh)
32 cv2.waitKey(0)
