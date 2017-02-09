import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.util.shape import view_as_windows


class Harris:
    def get_harris_image(self, image, blockSize=2, sobel_ksize=3, harris_param=0.05):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv2.cornerHarris(gray, blockSize, sobel_ksize, harris_param)

        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        return dst

    def get_harris_coords(self, harrisim, min_dist=5, threshold=0.01):
        """ Return corners from a Harris response image
        min_dist is the minimum number of pixels separating
        corners and image boundary. """

        # find top corner candidates above a threshold
        corner_threshold = harrisim.max() * threshold
        harrisim_t = (harrisim > corner_threshold) * 1
        coords = np.array(harrisim_t.nonzero()).T

        candidate_values = [harrisim[c[0], c[1]] for c in coords]
        # sort candidates
        index = np.argsort(candidate_values)
        # store allowed point locations in array
        allowed_locations = np.zeros(harrisim.shape)
        allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
        # select the best points taking min_distance into account
        filtered_coords = []
        for i in index:
            if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
                filtered_coords.append(coords[i])
                allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
                (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

        len(filtered_coords)
        return filtered_coords

    def get_descriptors(self, image, filtered_coords, wid=5):
        """ For each point return pixel values around the point
        using a neighbourhood of width 2*wid+1. (Assume points are
        extracted with min_distance > wid). """

        desc = []
        for coords in filtered_coords:
            patch = image[coords[0] - wid:coords[0] + wid + 1,
                    coords[1] - wid:coords[1] + wid + 1].flatten()
            desc.append(patch)
        return desc

    def draw_harris_points(self, image, harris_coords):
        """
        Shows the image marking the harris coordinates passed as parameters with circles
        :param image:
        :param harris_coords:
        :return:
        """
        cx = [x[0] for x in harris_coords]
        cy = [x[1] for x in harris_coords]
        # demasiado pequeno, no se ve
        image[cx, cy] = [0, 255, 0]

        for point in harris_coords:
            cv2.circle(image, (point[1], point[0]), 2, (0, 255, 0), -1)
        return

    def show_harris_corners(self, image, blockSize=5, sobel_ksize=5, threshold=0.03):
        """
        Calculates harris corner keyfeatures and marks them in the image
        """
        harrisim = self.get_harris_image(image, blockSize=blockSize, sobel_ksize=sobel_ksize)
        harris_coords = self.get_harris_coords(harrisim, threshold=threshold)
        self.draw_harris_points(image, harris_coords)
        return

    def match(self, desc1, desc2, threshold=0.5):
        """ For each corner point descriptor in the first image,
        select its match to second image using
        normalized cross correlation. """
        n = len(desc1[0])
        # pair-wise distances
        d = -np.ones((len(desc1), len(desc2)))
        for i in range(len(desc1)):
            for j in range(len(desc2)):
                d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
                d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
                ncc_value = sum(d1 * d2) / (n - 1)
                if ncc_value > threshold:
                    d[i, j] = ncc_value
        ndx = np.argsort(-d)
        matchscores = ndx[:, 0]
        return matchscores

    def match_twosided(self, desc1, desc2, threshold=0.5):
        """ Two-sided symmetric version of match(). """
        matches_12 = self.match(desc1, desc2, threshold)
        matches_21 = self.match(desc2, desc1, threshold)
        ndx_12 = np.where(matches_12 >= 0)[0]
        # remove matches that are not symmetric
        for n in ndx_12:
            if matches_21[matches_12[n]] != n:
                matches_12[n] = -1
        return matches_12


def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from 'match()'),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))
    plt.imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plt.plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
            plt.axis('off')


def get_pyramids(image, min_h, min_w, scale=1.5):
    # rows, cols, dim = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=scale))
    #
    # composite_image = np.zeros((rows, cols + cols / 2, 3), dtype=np.double)
    #
    # composite_image[:rows, :cols, :] = pyramid[0]
    #
    valid_pyd = []

    for p in pyramid[0:]:
        n_rows, n_cols = p.shape[:2]
        #
        if n_rows < min_h or n_cols < min_w:
            return valid_pyd
        else:
            valid_pyd.append(p)

    return


def test_harris_params(img):
    harris = Harris()

    images = []
    for b in range(4, 8):
        for s in range(3, 9, 2):
            item_img = img.copy()

            harrisim = harris.get_harris_image(item_img, blockSize=b, sobel_ksize=s)
            harris_coords = harris.get_harris_coords(harrisim, threshold=0.01)
            harris.draw_harris_points(item_img, harris_coords)

            images.append([item_img, "Blocksize=%s sobel=%s" % (b, s)])

    fig = plt.figure()
    i = 1
    idx = 430
    for img in images:
        # plt.subplot(3,3,i)
        ax = fig.add_subplot(4, 3, i)
        ax.imshow(img[0])
        ax.set_title(img[1])
        ax.set_axis_off()
        i += 1

    plt.axis("off")

    plt.show()


class OrbDetector():
    orb = None

    def __init__(self):
        self.orbt = cv2.ORB_create(nfeatures=200, scoreType=cv2.ORB_HARRIS_SCORE)
        self.orbi = cv2.ORB_create(nfeatures=1000, scoreType=cv2.ORB_HARRIS_SCORE)

    def camera_keypoints(self):
        cap = cv2.VideoCapture(0)

        while (1):
            # Take each frame
            _, frame = cap.read()
            img = self.show_keypoints(frame)

            cv2.imshow('frame', img)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    def _calculate_object_scale(self, img, img_kp_lst):
        """
        Given an image and its keypoints, determines the scale.
        The relation between the image size and the keypoint bounding-box
        :return:
        """
        points = [(int(kp.pt[0]), int(kp.pt[1])) for kp in img_kp_lst]
        points = np.array(points)
        x, y, w, h = cv2.boundingRect(points)
        margin = 20
        if len(img.shape) > 2:
            img_h, img_w = img.shape[:-1]
        else:
            img_h, img_w = img.shape[:2]

        return (float(img_w) / w, float(img_h) / h)


    def pre_process(self, image):
        # converto to GRAYSCALE if needed
        img = image
        if len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        return img

    def show_template_matches_knn(self, q_image, t_image):

        kp1, des1 = self.orbt.detectAndCompute(q_image, None)
        kp2, des2 = self.orbi.detectAndCompute(t_image, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        if type(des1) == type(None) or type(des2) == type(None):
            return t_image,0
        matches = bf.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        MIN_MATCH_COUNT = 5
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if type(mask) == type(None) or type(M) == type(None):
                img2 = t_image
                matchesMask = None
            else:
                matchesMask = mask.ravel().tolist()
                h, w = q_image.shape[0],q_image.shape[1]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                img2 = cv2.polylines(t_image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        else:
            # print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
            matchesMask = None
            img2 = t_image

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

        img3 = cv2.drawMatches(q_image, kp1, img2, kp2, good, None, **draw_params)

        return img3, 0


    def show_template_matches_mod(self, query_img, train_img):

        kp_qry, des_qry = self.orbt.detectAndCompute(query_img, None)
        kp_trn, des_trn = self.orbi.detectAndCompute(train_img, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # devuelve los resultados mas relevantes
        # query: objeto que buscamos, train: imagen sobre la que buscamos
        matches = bf.match(des_qry, des_trn)
        matches = sorted(matches, key=lambda x: x.distance)

        # matches = sorted(matches, key=lambda x:x.distance)
        # keep just more relevant points DMatch.distance
        matches, lst_objIdx, lst_imgIdx, distances = zip(
            *[(m, m.queryIdx, m.trainIdx, m.distance) for m in matches if m.distance < 100])

        print "distances: %s, %s" % (min(distances), max(distances))

        # marcamos los tres puntos con mayor y menor distancia
        distances = sorted(distances)
        NUM_RELEVANT_POINTS = 10
        distances = distances[:NUM_RELEVANT_POINTS] + distances[-NUM_RELEVANT_POINTS:]

        # sacar puntos a marcar
        lst_remarkable = []
        matches = sorted(matches, key=lambda x: x.distance)
        aux_dist = list(distances)
        for m in matches:
            if m.distance in aux_dist:
                aux_dist.remove(m.distance)
                lst_remarkable.append(m.trainIdx)

        img3 = cv2.drawMatches(query_img, kp_qry, train_img, kp_trn, matches[:40], train_img, flags=2)

        # mark points
        points = [(int(kp_trn[x].pt[0]), int(kp_trn[x].pt[1])) for x in lst_imgIdx]
        points = np.array(points)
        x, y, w, h = cv2.boundingRect(points)

        relevant_kp_qry = [kp_qry[i] for i in lst_objIdx]
        hor_scale, ver_scale = self._calculate_object_scale(query_img, relevant_kp_qry)
        margins = [int(w * hor_scale / 2), int(h * ver_scale / 2)]
        cv2.rectangle(img3, (x - margins[0], y - margins[1]), (x + w + margins[0], y + h + margins[1]), (0, 255, 0), 2)

        # marcar puntos relevantes
        points = [(int(kp_trn[x].pt[0]), int(kp_trn[x].pt[1])) for x in lst_remarkable]
        i = 0
        color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        print distances
        for p in points:
            cv2.circle(img3, (p[0], p[1]), 10, color, -1)
            distance = matches[i].distance
            if i >= NUM_RELEVANT_POINTS:
                distance = matches[-(i - NUM_RELEVANT_POINTS + 1)].distance
            cv2.putText(img3, str(distance), (p[0] + 12, p[1]), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if i >= NUM_RELEVANT_POINTS:
                color = (255, 0, 0)
            i += 1

        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        return img3

    def show_keypoints(self, image):
        kp1, des1 = self.orbi.detectAndCompute(image, None)
        img = image.copy()
        cv2.drawKeypoints(image, keypoints=kp1, outImage=img)
        return img

    def calculate_inc_mean(self, mean_matches,num_elements, a):
        m = mean_matches + (a)/(num_elements+1)
        return m

    def camera_detect(self, object_img):
        cap = cv2.VideoCapture(0)

        mean = 0
        num_elements = 0

        queryImage = self.pre_process(object_img)

        while (1):
            # Take each frame
            _, frame = cap.read()

            trainingImage = self.pre_process(frame)
            img, num_matches = self.show_template_matches_knn(object_img,frame)
            mean = self.calculate_inc_mean(mean,num_elements, num_matches)
            num_elements +=1

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img, 'matches: ' +str(mean), (10, 10), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', img)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()

    def init_detector(self):
        # arrancamos camara
        # seleccionamos un area
        # calculamos keypoints
        # aplicamos
        pass


def test_harris():
    harris = Harris()

    template = cv2.imread('../images/test/templates/tplana.jpg')
    image = cv2.imread('../images/test/t1.jpg')
    img = template

    test_harris_params(img)


def test_orb_camera_detector(template_path):
    orb = OrbDetector()
    template = cv2.imread(template_path)
    orb.camera_detect(template)

def test_orb():
    orb = OrbDetector()

    template = cv2.imread('../images/test2/template.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('../images/test2/t4.jpg', cv2.IMREAD_GRAYSCALE)

    # template = cv2.bilateralFilter(template, 7, 41, 41)
    # image = cv2.bilateralFilter(image, 7, 41, 41)

    image = cv2.equalizeHist(image)
    template = cv2.equalizeHist(template)

    img = orb.show_template_matches_knn(template, image)
    plt.imshow(img), plt.show()
    # orb.camera_keypoints()



def test_orb():
    orb = OrbDetector()

    template = cv2.imread('../images/test2/template.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('../images/test2/t4.jpg', cv2.IMREAD_GRAYSCALE)

    img = orb.show_template_matches_knn(template, image)
    plt.imshow(img), plt.show()
    # orb.camera_keypoints()

def test_akaze_camera_detector(template_path):
    orb = OrbDetector()
    orb.orbi = cv2.AKAZE_create()
    orb.orbi = cv2.AKAZE_create()

    template = cv2.imread(template_path)
    orb.camera_detect(template)

def kaze_match(query, train):
    # load the image and convert it to grayscale

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(query, None)
    (kps2, descs2) = detector.detectAndCompute(train, None)

    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs1, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3 = cv2.drawMatchesKnn(query, kps1, train, kps2, good[1:20], None, flags=2)
    cv2.imshow("AKAZE matching", im3)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_orb_camera_detector('../../resources/images/book-cover.jpg')

    # query= cv2.imread('../images/test2/template.jpg', cv2.IMREAD_GRAYSCALE)
    # train = cv2.imread('../images/test2/t1.jpg', cv2.IMREAD_GRAYSCALE)

    # kaze_match(query, train)





