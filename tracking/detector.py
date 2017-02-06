import os

import cv2
import skimage.util as util
from skimage import io
from skimage.transform import pyramid_gaussian
from skimage.util import pad
from skimage.util.shape import view_as_windows

import config as cfg
from tracking.lab.keypoint_detectors import Harris


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
        if n_rows < min_h or n_cols< min_w:
            return valid_pyd
        else:
            valid_pyd.append(p)

    return valid_pyd

def testImage(imagePath, decisionThreshold = cfg.decision_threshold, applyNMS=True):

    # file = open(cfg.modelPath, 'r')
    # svc = pickle.load(file)

    image = io.imread(imagePath, as_grey=True)
    image = util.img_as_ubyte(image) #Read the image as bytes (pixels with values 0-255)

    rows, cols = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=cfg.downScaleFactor))

    scale = 0
    boxes = None
    scores = None
    subimagenes = 0

    for p in pyramid[0:]:
        #We now have the subsampled image in p

        #Add padding to the image, using reflection to avoid border effects
        if cfg.padding > 0:
            p = pad(p,cfg.padding,'reflect')



        try:
            views = view_as_windows(p, cfg.window_shape, step=cfg.window_step)
        except ValueError:
            #block shape is bigger than image
            break

        num_rows, num_cols, width, height = views.shape
        for row in range(0, num_rows):
            for col in range(0, num_cols):

                #Get current window
                subImage = views[row, col]

                # cv2.imwrite("%s/im_%s_%s.jpg" % (cfg.TMP_FOLDER, row, col), subImage)
                if subimagenes <20 and scale > 10:
                    cv2.imshow("im_%s_%s" % (row, col), subImage)

                    cv2.imwrite("/tmp/im_%s_%s.png"% (row, col),subImage)
                    subimagenes +=1
        scale += 1

    # if applyNMS:
    #     #From all the bounding boxes that are overlapping, take those with maximum score.
    #     boxes, scores = nms.non_max_suppression_fast(boxes, scores, cfg.nmsOverlapThresh)

    return boxes, scores


        
def impath(filename):
    path1  = "../images/"
    path2  = "images/"

    if os.path.exists(os.path.join(path1, filename)):
        return os.path.join(path1, filename)
    else:
        return os.path.join(path2,filename)


def process_capture(func_img, *args):
    cap = cv2.VideoCapture(0)

    while (1):
        # Take each frame
        _, frame = cap.read()

        func_img(frame, *args)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def init_tracking():

    cap = cv2.VideoCapture(0)

    filename = impath('grua-clip.jpg')
    template = cv2.imread(filename)
    tpt_h, tpt_w, ch = template.shape
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    #

    while(1):
        # Take each frame
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # harris = Harris()
        #
        # harrisim = harris.get_harris_image(frame)
        # harris_coords = harris.get_harris_coords(harrisim)
        # harris.draw_harris_points(frame, harris_coords)
        # harris_desc = harris.get_descriptors(frame, harris_coords)
        pyramids = get_pyramids(frame, tpt_h,tpt_w,scale=1.5)
        print len(pyramids)

        for p in pyramids:
            print "Encontradas %s piramides"%(len(p))
            views = view_as_windows(p, (128, 64), step=32)
            num_rows, num_cols, width, height = views.shape

            for row in range(0, num_rows):
                for col in range(0, num_cols):
                    # Get current window
                    subImage = views[row, col]
                    w,h = subImage.shape

        cv2.imshow('frame',frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()



def process_capture():
    hrr = Harris()
    cap = cv2.VideoCapture(0)
    while (1):
        # Take each frame
        _, frame = cap.read()
        hrr.show_harris_corners(frame,threshold=0.001,sobel_ksize=7, blockSize=6)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

process_capture()