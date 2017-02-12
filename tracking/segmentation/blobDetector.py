import utils.imutils as imu


def opencv_blob_detection():
    """
    # based on https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    :return:
    """
    # Standard imports
    import cv2
    import numpy as np;

    # Read image
    im_path = imu.load_image("toy.jpg")
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)

    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 500;
    #
    # # Filter by Area.
    # params.filterByArea = True
    # params.minArea = 1500
    #
    # # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.1
    #
    # # Filter by Convexity
    params.filterByConvexity = False
    # params.minConvexity = 0.87
    #
    # # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01


    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)

    myblobs = []
    # Detect blobs.
    keypoints = detector.detect(im,myblobs)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

def scikit_blob_detection():
    from matplotlib import pyplot as plt
    from skimage import data
    from skimage.feature import blob_dog, blob_log, blob_doh
    from math import sqrt
    from skimage.color import rgb2gray
    from skimage import io

    im_path = imu.load_image("toy.jpg")
    image = io.imread(im_path)

#    image = data.hubble_deep_field()[0:500, 0:500]
    image_gray = rgb2gray(image)

    blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
    axes = axes.ravel()
    for blobs, color, title in sequence:
        ax = axes[0]
        axes = axes[1:]
        ax.set_title(title)
        ax.imshow(image, interpolation='nearest')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax.add_patch(c)

    plt.show()

if __name__ == '__main__':
    scikit_blob_detection()