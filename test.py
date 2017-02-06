# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# import the necessary packages
import numpy as np
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = len(np.unique(clt.labels_))
    (hist, _) = np.histogram(clt.cluster_centers_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def get_histo(pxls):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = len(np.unique(clt.labels_))
    (hist, _) = np.histogram(pxls)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

def cvexample(img):
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    print(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# construct the argument parser and parse the arguments




# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread("images/grua.jpg")

cvexample(image)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # show our image
# plt.figure()
# plt.axis("off")
# plt.imshow(image)
#
# # reshape the image to be a list of pixels
# image = image.reshape((image.shape[0] * image.shape[1], 3))
#
# # cluster the pixel intensities
# print "entregandno......"
# clt = KMeans(n_clusters = 8)
# print "entregandno......2"
# clt.fit(image)
# print "entregandno......3"
#
# # build a histogram of clusters and then create a figure
# # representing the number of pixels labeled to each color
# # hist = centroid_histogram(clt)
# (hist, _) = np.histogram(image)
#
# print "entregandno......4"
# bar = plot_colors(hist, clt.cluster_centers_)
#
# # show our color bart
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
# plt.show()