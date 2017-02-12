# Import the necessary packages
import cv2
import numpy as np


def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Return the translated image
    return shifted


def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


def resize(image, width=None, height=None, porc=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None and porc is None:
        return image

    # check to see if the width is None
    if porc is not None:
        # rescale image using a % applied to both width and height
        dim = (int(w * porc), int(h * porc))

    elif width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def imshow(im, text=" "):
    cv2.imshow(text, im)
    cv2.waitKey(0)


def load_image(img_name, folder=None):
    import os
    """
    Load an imagen from resoures/images folder
    :param img_name:
    :return:
    """
    if not folder:
        base_path = os.path.join(os.path.dirname(__file__), '../resources/images', img_name)
    else:
        base_path = os.path.join(os.path.dirname(__file__), '../..', folder, img_name)

    return os.path.normpath(base_path)
