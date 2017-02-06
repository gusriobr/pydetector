import cv2
import matplotlib.pyplot as plt
import numpy as np


class FeatureDetector():
    query_image_proc = None # query image processed
    query_keypts = None
    query_descs = None

    detector_t = None
    descriptor_t = None
    detector_i = None
    descriptor_i = None
    MIN_MATCH_COUNT = 5

    def __init__(self, detector):
        self.MIN_MATCH_COUNT = 5
        self.alg = detector.upper()
        if detector == "orb":
            self.detector_q = cv2.ORB_create(nfeatures=200, scoreType=cv2.ORB_HARRIS_SCORE)
            self.descriptor_q = self.detector_q
            self.detector_i = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE)
            self.descriptor_i = self.detector_i
        if detector == "brisk":
            self.detector_q = cv2.BRISK_create()
            self.descriptor_q = self.detector_q
            self.detector_i = cv2.BRISK_create()
            self.descriptor_i = self.detector_i
        if detector == "brisk-freak":
            self.detector_q = cv2.BRISK_create()
            self.descriptor_q = cv2.xfeatures2d.FREAK_create()
            self.detector_i = cv2.BRISK_create()
            self.descriptor_i = cv2.xfeatures2d.FREAK_create()

    def __create_img_label(self, image, num):
        width = image.shape[1]
        line_heigth = 30
        if len(image.shape) == 2:
            label = np.zeros((2*line_heigth,width), dtype=np.uint8)
        else:
            label = np.zeros((2 * line_heigth, width,3), dtype=np.uint8)
        cv2.putText(label, self.alg, (5, line_heigth-2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        cv2.putText(label, "#kpts:"+str(num), (5, 2*(line_heigth-2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255)
        image = np.vstack((image,label))
        return image


    def __basic_preprocess(self,image):
        # convert to GRAYSCALE if needed
        return image
        # img = image
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # img = cv2.equalizeHist(img)
        # return img

    def pre_process_query_image(self, image):
        img = self.__basic_preprocess(image)
        # calculate keypoints and descriptors
        self.query_keypts = self.detector_q.detect(img, None)
        self.query_keypts, self.query_descs = self.descriptor_q.compute(img, self.query_keypts)

        # show number of keypoints
        img = self.__create_img_label(img, len(self.query_keypts))

        # draw keypoints in imagen. this turns the image back into RBG space
        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=self.query_keypts,
                                flags=0, color=(51, 163, 236))

        self.query_image_proc = img
        return self.query_image_proc

    def pre_process_train_image(self, image):
        return self.__basic_preprocess(image)

    def show_template_matches_knn(self, t_image):

        # kp2, des2 = self.orbi.detectAndCompute(t_image, None)
        kp2 = self.detector_i.detect(t_image, None)
        kp2, des2 = self.descriptor_i.compute(t_image, kp2)


        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#         bf = cv2.BFMatcher(cv2.NORM_L2SQR )
        if type(self.query_descs) == type(None) or type(des2) == type(None):
            return t_image
        matches = bf.knnMatch(self.query_descs, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)


        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([self.query_keypts[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            img2, matchesMask = self.tag_object_by_mass_center(dst_pts, src_pts, t_image)

        else:
            # print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            matchesMask = None
            img2 = t_image

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

        img3 = cv2.drawMatches(self.query_image_proc, self.query_keypts, img2, kp2, good, None, **draw_params)

        return img3

    def tag_object_by_homography(self, dst_pts, src_pts, t_image):
        """"
        Marks detected object using homography
        """
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if type(mask) == type(None) or type(M) == type(None):
            img2 = t_image
            matchesMask = None
        else:
            matchesMask = mask.ravel().tolist()
            h, w = self.query_image_proc.shape[0], self.query_image_proc.shape[1]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            img2 = cv2.polylines(t_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        return img2, matchesMask

    def tag_object_by_convex_hull(self, dst_pts, src_pts, t_image):
        """"
        Marks detected object using convex hull
        """
        hull = cv2.convexHull(dst_pts)

        img2 = cv2.polylines(t_image, [np.int32(hull)], True, 255, 3, cv2.LINE_AA)
        matchesMask = None
        return img2, matchesMask

    def tag_object_by_mass_center(self, dst_pts, src_pts, t_image):
        """"
        Marks detected object using convex hull
        """
        M = cv2.moments(dst_pts)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(t_image, (cX, cY), 7, (255, 255, 255), -1)
            # draw contour
            w,h = self.query_image_proc.shape[:2]
            w, h = w/2,h/2
            cv2.rectangle(t_image, (cX-w, cY-h), (cX+w, cY+h), (0, 255, 0), 3)

        except:
            pass
        img2 = t_image
        matchesMask = None

        return img2, matchesMask



    def set_query_image(self, image):
        self.query_image_proc = self.pre_process_query_image(image)

    def find_query_image(self, image):
        train_img = self.pre_process_train_image (image)
        return self.show_template_matches_knn(train_img)

