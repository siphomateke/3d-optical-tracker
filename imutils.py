import cv2
import numpy as np


def resize_img(img, width=-1, height=-1):
    if height == -1 and width == -1:
        raise TypeError("Invalid arguments. Width or height must be provided.")
    h = img.shape[0]
    w = img.shape[1]
    if height == -1:
        aspect_ratio = float(w) / h
        new_height = int(width / aspect_ratio)
        return cv2.resize(img, (width, new_height))
    elif width == -1:
        aspect_ratio = h / float(w)
        new_width = int(height / aspect_ratio)
        return cv2.resize(img, (new_width, height))


def rotate_img(image, angle, bounds=False):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    if bounds:
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
    else:
        nW = image.shape[1]
        nH = image.shape[0]

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
