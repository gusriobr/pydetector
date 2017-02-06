import cv2
import numpy as np

def imshow(image):
	cv2.imshow("Gaussian Thresh", image)
	cv2.waitKey(0)

def create_image_mask(img, color_boundaries):

    rb = color_boundaries[0]
    gb = color_boundaries[1]
    bb = color_boundaries[2]

    # img2 = img[(img[:, 0] <= bb[1]) & (img[:, 0] > bb[0])]
    img_b = img[:,:,0]
    img_g = img[:,:,1]
    img_r = img[:,:,2]
    img_b = (img_b <= bb[1]) & (img_b > bb[0])
    img_g = (img_g <= gb[1]) & (img_g > gb[0])
    img_r = (img_r <= rb[1]) & (img_r > rb[0])
    w,h,c = img.shape
    filter = np.zeros([w, h, c],dtype="uint8")

    filter[:,:,0] = img_b
    filter[:,:,1] = img_g
    filter[:,:,2] = img_r

    return filter.astype("uint8")


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

def get_ppal_colors(image):
    # import the necessary packages
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img2 = image.reshape((img2.shape[0] * img2.shape[1], 3))
    clt = KMeans(n_clusters=4)
    clt.fit(image)

    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)


if __name__ == '__main__':
    image = cv2.imread("../images/grua.jpg")

    mask = create_image_mask(image, ([100,200],[100,200],[100,200]))

    imshow(image*mask)
