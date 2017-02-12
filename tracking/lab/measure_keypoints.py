""" How to determine how good are the keypoints detected by a feature tracker?"""

""""
1.- Obtain keypoints, flip and rotate image and check the keypoints against the second version
of the image
"""

import os
import time

import cv2

import utils.imutils as imu
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
    # img_path = os.path.join(os.path.dirname(__file__), '../../resources/images/book_cover.jpg')
    base_img_path = os.path.join(os.path.dirname(__file__), '../../resources/images')
    img_list = ["book_cover.jpg", "toy.jpg"]
    img_list = [os.path.join(base_img_path, x) for x in img_list]

    list = [(x, y, z)
            for z in img_list
            for x in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6]
            for y in ["orb", "brisk"]
            ]

    # header
    print "Image\tMethod\tscale%\ttrain kps\tqry kps\tmatches\tgood\trate\ttime(ms)"
    for porc, method, img_path in list:
        image = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        start_time = time.time()
        img, stats = measure_kpts_relavance(image, porc, method)
        elapsed_time = time.time() - start_time
        print "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f%%\t%.2f" % (
            img_name, method, porc, stats.num_train_kpts, stats.num_query_kpts, stats.num_matches,
            stats.num_good_matches,
            stats.rate_matches,elapsed_time*1000)
