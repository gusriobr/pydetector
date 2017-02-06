""" How to determine how good are the keypoints detected by a feature tracker?"""

""""
1.- Obtain keypoints, flip and rotate image and check the keypoints against the second version
of the image
"""

import cv2
import os
import tracking.imutils as imu
from tracking.feature_based.feature_tracker import FeatureDetector


def measure_kpts_relavance(image, porc, method):
    im2 = imu.resize(image, porc=porc)
    im2 = imu.rotate(im2, 90)
    # im2 = cv2.flip(im2,1)

    # find keypoints
    fd = FeatureDetector(method)
    # set original imagen as query imagen
    fd.set_query_image(image)
    # get keypoints
    img = fd.find_query_image(im2)
    return img, fd.stats


if __name__ == '__main__':
    img_path = os.path.join(os.path.dirname(__file__), '../../resources/images/book_cover.jpg')
    image = cv2.imread(img_path)

    list = [(x, y) for x in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6] for y in ["orb", "brisk"]]
    for porc, method in list:
        img, stats = measure_kpts_relavance(image, porc, method)
        print "method %s, porc %s: %s" % (method, porc, stats)
