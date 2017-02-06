import cv2
import numpy    as    np
from matplotlib import pyplot as plt

from tracking.feature_tracker import FeatureDetector


class CameraTracker(object):
    detector = None
    """
        Tracker object must implement two methods
        - set_query_image(np_array): None
        - find_query_image(image): image
    """

    def __init__(self):
        #	Initialize	the	video	capture	object
        #	0	->	indicates	that	frame	should	be	captured
        #	from	webcam
        self.cap = cv2.VideoCapture(0)
        #	Capture	the	frame	from	the	webcam
        ret, self.frame = self.cap.read()
        #	Downsampling	factor	for	the	input	frame
        self.scaling_factor =1
        self.frame = cv2.resize(self.frame, None, fx=self.scaling_factor,
                                fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
        cv2.namedWindow('Object	Tracker')
        cv2.setMouseCallback('Object	Tracker', self.mouse_event)
        self.selection = None
        self.drag_start = None
        self.tracking_state = 0


    def extract_foreground(self, img, selection_tuple):
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        x0,y0,x1,y1 = selection_tuple
        rect = (x0, y0, x1-x0, y1-y0)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = img * mask2[:, :, np.newaxis]
        # recortar imagen
        img = img[y0:y1, x0:x1]
        return img


    #	Method	to	track	mouse	events
    def mouse_event(self, event, x, y, flags, param):
        x, y = np.int16([x, y])
        #	Detecting	the	mouse	button	down	event
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
            # return
        if self.drag_start:
            # if event == cv2.EVENT_LBUTTONUP:
            # if event == cv2.EVENT_RBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                # self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1
                    if self.detector:
                        # extract selected imagen
                        x0, y0, x1, y1 = self.selection
                        #img = self.extract_foreground(self.frame, self.selection)
                        img = self.frame[y0:y1, x0:x1]
                        self.detector.set_query_image(img)


    def start_tracking(self):
        #	Iterate	until	the	user	presses	the	Esc	key
        while True:
        # Capture	the	frame	from	webcam
            ret, self.frame = self.cap.read()
            #	Resize	the	input	frame
            self.frame = cv2.resize(self.frame, None,
                                    fx=self.scaling_factor,
                                    fy=self.scaling_factor,
                                    interpolation=cv2.INTER_AREA)

            view = self.frame

            if self.selection:
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1 - x0, y1 - y0)
                # on selection end
                BORDER = 2
                cv2.rectangle(self.frame, (x0-BORDER, y0-BORDER), (x1+BORDER, y1+BORDER), (0, 255, 0), 2)

            if self.tracking_state == 1:
                self.selection = None
                #	Compute	the	histogram	back	projection
                view = self.detector.find_query_image(view)

            cv2.imshow('Object	Tracker', view)
            c = cv2.waitKey(5)
            if c == 27:
                break

    cv2.destroyAllWindows()

class SimpleDetector():
    query_image = None

    def set_query_image(self, image):
        plt.imshow(image),plt.show()
        self.query_image = image

    def find_query_image(self, image):
        return image





if __name__ == '__main__':
    ctracker = CameraTracker()
    #ctracker.detector = SimpleDetector()
    ctracker.detector = FeatureDetector("brisk")

    ctracker.start_tracking()
