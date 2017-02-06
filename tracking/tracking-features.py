import numpy as np
import cv2

import matplotlib.pyplot as plt
img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


def load_template():
    tem_img = cv2.imread("../images/grua-black.jpg")
    # tem_img = cv2.GaussianBlur(tem_img,(3,3),0)
    # tem_img = cv2.cvtColor(tem_img,cv2.COLOR_BGR2GRAY)
    return tem_img


def feature_detection(image, template):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = image.copy()
    img3 = cv2.drawMatches(template,kp1,image,kp2,matches[:10], img3, flags=2)

    cv2.imshow("matching",img3)






cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()

    template = load_template()
    feature_detection(template,frame)


    cv2.imshow('frame',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
