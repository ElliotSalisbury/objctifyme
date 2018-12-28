import os

import cv2
import dlib
import numpy as np

# initialize dlib detector
scriptFolder = os.path.dirname(os.path.realpath(__file__))
DLIB_SHAPEPREDICTOR_PATH = os.environ['DLIB_SHAPEPREDICTOR_PATH']

LEFT_EYE = list(range(36,42))
RIGHT_EYE = list(range(42,48))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPEPREDICTOR_PATH)

def getLandmarks(im, rect=None):
    if rect is None:
        rects = detector(im, 1)
        if len(rects) == 0:
            raise Exception("No face could be detected in the image.")
        if len(rects) > 1:
            raise Exception("Multiple faces were detected, we currently only support images of a single face.")

        rect = rects[0]

    landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    landmarks = np.array([[p[0, 0], p[0, 1]] for p in landmarks])

    return landmarks


def getNormalizingFactor(landmarks):
    hull = cv2.convexHull(landmarks)
    return np.sqrt(cv2.contourArea(hull))

# def featuresFromLandmarks(landmarks):
#     normalizingTerm = getNormalizingFactor(landmarks)
#     normLandmarks = landmarks / normalizingTerm
#
#     faceFeatures = normLandmarks[faceLines[:, 0]] - normLandmarks[faceLines[:, 1]]
#     faceFeatures = np.linalg.norm(faceFeatures, axis=1)
#     return faceFeatures

# def getFaceFeatures(im):
#     landmarks = getLandmarks(im)
#
#     faceFeatures = featuresFromLandmarks(landmarks)
#
#     return landmarks, faceFeatures
