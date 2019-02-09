import cv2
import numpy as np


def image_resize(image, max_width=512, max_height=512, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # check to see if the width is None
    if h > w:
        # calculate the ratio of the height and construct the
        # dimensions
        r = max_height / float(h)
        dim = (int(w * r), max_height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = max_width / float(w)
        dim = (max_width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def ensure_max_image_size(image, max_size=4096, inter=cv2.INTER_AREA):
    if image.shape[0] > max_size or image.shape[1] > max_size:
        image = image_resize(image, max_width=max_size, max_height=max_size, inter=inter)
    return image


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    dstTri = np.array([[0, 0], [w, 0], [0, h]], dtype=np.float32)
    srcTri = np.array([(np.matrix(M) * np.append(p, 1)[:, np.newaxis]) for p in dstTri], dtype=np.float32).squeeze()

    invM = cv2.getAffineTransform(srcTri, dstTri)

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), M, invM
