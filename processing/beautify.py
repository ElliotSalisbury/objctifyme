import math
import os.path
import sys

import cv2
import dlib
import numpy as np
import tensorflow as tf

from processing.Beautifier.face3D.warpFace3D import warpFace3D, projectMeshTo2D, ALL_FACE_LANDMARKS

os.environ.setdefault("DLIB_SHAPEPREDICTOR_PATH", "./processing/shape_predictor_68_face_landmarks.dat")
os.environ.setdefault("EOS_DATA_PATH", "./processing/share")

from processing.Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING
from processing.Beautifier.faceFeatures import getLandmarks, LEFT_EYE, RIGHT_EYE

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings

sys.path.append('./processing/kaffe')
sys.path.append('./processing/ResNet')
from processing.ResNet.ThreeDMM_shape import ResNet_101 as resnet101_shape
from processing.ResNet.ThreeDMM_expr import ResNet_101 as resnet101_expr
from processing.im_utils import image_resize, rotate_bound, ensure_max_image_size

tf.logging.set_verbosity(tf.logging.INFO)

image_size = 227
num_gpus = 0
batch_size = 1

if num_gpus == 0:
    dev = '/cpu:0'
elif num_gpus == 1:
    dev = '/gpu:0'
else:
    raise ValueError('Only support 0 or 1 gpu.')

tf.device(dev)

# Global parameters
_tmpdir = './tmp/'  # save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet
output_proc = 'output_preproc.csv'  # save intermediate image list
output_val = 'output_preproc_val.csv'  # save intermediate image list
factor = 0.25  # expand the given face bounding box to fit in the DCCNs
_alexNetSize = 227

DLIB_SHAPEPREDICTOR_PATH = os.environ.get("DLIB_SHAPEPREDICTOR_PATH")
MEDIA_ROOT = settings.MEDIA_ROOT
MAX_ALIGN_ATTEMPTS = 10
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPEPREDICTOR_PATH)

sess = None
x = None
hot_score = None
hot_grad = None
fc1ls = None
fc1le = None

def init_network():
    global sess, x, hot_score, hot_grad, fc1ls, fc1le

    if sess is None:
        # Get training image mean for Shape CNN
        mean_image_shape = np.load('./processing/Shape_Model/3DMM_shape_mean.npy', fix_imports=True,
                                   encoding='bytes')  # 3 x 224 x 224
        mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]

        # set up the network
        x = tf.placeholder(tf.float32, [batch_size, None, None, 3])

        ###################
        # Shape CNN
        ###################
        x2 = tf.image.resize_bilinear(x, tf.constant([224, 224], dtype=tf.int32))
        x2 = tf.cast(x2, 'float32')
        x2 = tf.reshape(x2, [batch_size, 224, 224, 3])

        # Image normalization
        mean = tf.reshape(mean_image_shape, [1, 224, 224, 3])
        mean = tf.cast(mean, 'float32')
        x2 = x2 - mean

        with tf.variable_scope('shapeCNN'):
            net_shape = resnet101_shape({'input': x2}, trainable=False)
            pool5 = net_shape.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [batch_size, -1])

            npzfile = np.load('./processing/ResNet/ShapeNet_fc_weights.npz', fix_imports=True, encoding='bytes')
            ini_weights_shape = npzfile['ini_weights_shape']
            ini_biases_shape = npzfile['ini_biases_shape']
            with tf.variable_scope('shapeCNN_fc1'):
                fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048, -1]), trainable=False, name='weights')
                fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=False, name='biases')
                fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)

            with tf.variable_scope('shapeCNN_fc2'):
                dense = tf.layers.dense(inputs=fc1ls, units=256, activation=tf.nn.relu)
                # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

            with tf.variable_scope('shapeCNN_fc3'):
                dense = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu)
                # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

            with tf.variable_scope('shapeCNN_output'):
                # Logits Layer
                hot_score = tf.layers.dense(inputs=dense, units=1)

            hot_grad = tf.gradients(hot_score, fc1ls)[0]

        ###################
        # Expression CNN
        ###################
        with tf.variable_scope('exprCNN'):
            net_expr = resnet101_expr({'input': x2}, trainable=True)
            pool5 = net_expr.layers['pool5']
            pool5 = tf.squeeze(pool5)
            pool5 = tf.reshape(pool5, [batch_size, -1])

            npzfile = np.load('./processing/ResNet/ExpNet_fc_weights.npz', fix_imports=True, encoding='bytes')
            ini_weights_expr = npzfile['ini_weights_expr']
            ini_biases_expr = npzfile['ini_biases_expr']
            with tf.variable_scope('exprCNN_fc1'):
                fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
                fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
                fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)

        # Add ops to save and restore all the variables.
        # Add ops to save and restore all the variables.
        init_op = tf.global_variables_initializer()
        all_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
        saver_ini_shape_net = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN')[:-6])
        saver_ini_expr_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(init_op)


def rotated_rect_from_landmarks(landmarks, M, im):
    rotated_rect_ps = np.array([(np.matrix(M) * np.append(p, 1)[:, np.newaxis]) for p in landmarks],
                               dtype=np.float32).squeeze()
    rotated_rect_ps = np.array([np.min(rotated_rect_ps, axis=0), np.max(rotated_rect_ps, axis=0)])
    rotated_rect_wh = rotated_rect_ps[1, :] - rotated_rect_ps[0, :]
    rotated_rect_c = rotated_rect_ps[0, :] + rotated_rect_wh / 2
    rotated_rect_nwh = (rotated_rect_wh * 1.1) / 2
    rotated_rect_ps = np.array([rotated_rect_c - rotated_rect_nwh, rotated_rect_c + rotated_rect_nwh])
    rotated_rect_ps = np.clip(rotated_rect_ps, 0, [im.shape[1], im.shape[0]])

    rotated_rect = dlib.rectangle(
        left=int(math.floor(rotated_rect_ps[0, 0])),
        top=int(math.floor(rotated_rect_ps[0, 1])),
        right=int(math.ceil(rotated_rect_ps[1, 0])),
        bottom=int(math.ceil(rotated_rect_ps[1, 1])))

    return rotated_rect


def align_face_vertical(image_orig, rect):
    # get the initial landmarks guess
    landmarks_orig = getLandmarks(image_orig, rect)

    # get the line from the left to right eye
    left_eye = np.mean(landmarks_orig[LEFT_EYE], axis=0)
    right_eye = np.mean(landmarks_orig[RIGHT_EYE], axis=0)

    # calculate the angle to make that horizontal
    dX = right_eye[0] - left_eye[0]
    dY = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dY, dX))
    rotate_angle = angle

    # rotate the image getting the transform and it's inverse
    rotated_im, M, invM = rotate_bound(image_orig, rotate_angle)
    rotated_rect = rotated_rect_from_landmarks(landmarks_orig, M, rotated_im)

    attempt = 0
    while abs(angle) > 4.0 and attempt < MAX_ALIGN_ATTEMPTS:
        # get the landmarks guess
        landmarks = getLandmarks(rotated_im, rotated_rect)

        # get the line from the left to right eye
        left_eye = np.mean(landmarks[LEFT_EYE], axis=0)
        right_eye = np.mean(landmarks[RIGHT_EYE], axis=0)

        # calculate the angle to make that horizontal
        dX = right_eye[0] - left_eye[0]
        dY = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dY, dX))
        rotate_angle += angle

        # rotate the image getting the transform and it's inverse
        rotated_im, M, invM = rotate_bound(image_orig, rotate_angle)
        rotated_rect = rotated_rect_from_landmarks(landmarks_orig, M, rotated_im)

        attempt += 1

    return rotated_im, M, invM, rotated_rect


def preprocess_image(im, rect=None, guess_rect=None):
    if rect is None:
        rects = detector(im, 1)
        if len(rects) == 1:
            rect = rects[0]
        elif guess_rect is not None:
            rect = guess_rect
        else:
            raise Exception("No face found in image")

    x1, y1, w, h = rect.left(), rect.top(), rect.width(), rect.height()
    max_side = max(w, h)
    border = (max_side * factor)

    center = np.array([x1 + w / 2, y1 + h / 2])
    bbox = np.zeros((4), dtype=np.int64)
    bbox[:2] = (center - max_side / 2 - border) + max_side
    bbox[2:] = (center + max_side / 2 + border) + max_side

    border_im = cv2.copyMakeBorder(im, max_side, max_side, max_side, max_side, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    face_im = border_im[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    face_im = cv2.resize(face_im, (_alexNetSize, _alexNetSize), interpolation=cv2.INTER_CUBIC)
    return face_im, rect


def decomposePose(mesh, pose, im):
    modelview = np.matrix(pose.get_modelview())
    proj = np.matrix(pose.get_projection())
    viewport = np.array([0, im.shape[0], im.shape[1], -im.shape[0]])

    modelview = modelview.tolist()
    projection = proj.tolist()
    viewport = viewport.tolist()

    ALL_FACE_MESH_VERTS = BFM_FACEFITTING.landmarks_2_vert_indices[ALL_FACE_LANDMARKS]
    ALL_FACE_MESH_VERTS = np.delete(ALL_FACE_MESH_VERTS, np.where(ALL_FACE_MESH_VERTS == -1)).tolist()
    verts2d = projectMeshTo2D(mesh, pose, im)
    convexHullIndexs = cv2.convexHull(verts2d.astype(np.float32), returnPoints=False)
    warpPointIndexs = convexHullIndexs.flatten().tolist() + ALL_FACE_MESH_VERTS
    indexs = warpPointIndexs

    return modelview, projection, viewport, indexs


def beautify_image(image_orig_full, gender):
    init_network()

    # load the image and resize it to something reasonable
    image_orig = ensure_max_image_size(image_orig_full)
    image_resized = image_resize(image_orig_full)

    h_scale = image_orig.shape[0] / image_resized.shape[0]
    w_scale = image_orig.shape[1] / image_resized.shape[1]

    # get the location of the face in the image
    rects = detector(image_resized, 1)

    rect_resized = rects[0]
    rect_orig = dlib.rectangle(
        left=int(math.floor(rect_resized.left() * w_scale)),
        top=int(math.floor(rect_resized.top() * h_scale)),
        right=int(math.ceil(rect_resized.right() * w_scale)),
        bottom=int(math.ceil(rect_resized.bottom() * h_scale)))

    # align the image so that the face is vertical
    image_rotated, M, invM, rotated_rect = align_face_vertical(image_orig, rect_orig)
    face_im, rect_rotated = preprocess_image(image_rotated, rect=None, guess_rect=rotated_rect)

    landmarks_rot = getLandmarks(image_rotated, rect_rotated)
    landmarks_orig = np.array(
        [(np.matrix(invM) * np.append(p, 1)[:, np.newaxis]) for p in landmarks_rot],
        dtype=np.int).squeeze()

    print("\trunning shape and expression networks")
    image = np.asarray(face_im)
    image = np.reshape(image, [1, image_size, image_size, 3])
    rating_orig, grad, shape_texture_orig, expression_orig = sess.run([hot_score, hot_grad, fc1ls, fc1le],
                                                                 feed_dict={x: image})
    shape_texture_orig = np.reshape(shape_texture_orig, [-1])
    expression_orig = np.reshape(expression_orig, [-1])
    grad = np.reshape(grad, [-1])

    shape_orig = shape_texture_orig[0:99]
    texture_orig = shape_texture_orig[99:]

    mesh_orig = BFM_FACEFITTING.getMeshFromShapeCeoffs(shape_orig, expression_orig, texture_orig)
    pose_orig = BFM_FACEFITTING.getPoseFromMesh(landmarks_orig, mesh_orig, image_orig)

    delta = np.zeros_like(grad)
    exaggeration = np.ones_like(grad)
    exaggeration[:99] = 1
    exaggeration[99:] = 1
    for i in range(6):
        delta += grad * exaggeration
        new_shape_texture = shape_texture_orig + delta
        shape = new_shape_texture[0:99]
        texture = new_shape_texture[99:]

        mesh_new = BFM_FACEFITTING.getMeshFromShapeCeoffs(shape, expression_orig, texture)
        warpedim = warpFace3D(image_orig, mesh_orig, pose_orig, mesh_new, accurate=False,
                              fitter=BFM_FACEFITTING)

        warped_rotated = cv2.warpAffine(warpedim, M, (image_rotated.shape[1], image_rotated.shape[0]))
        face_im, _ = preprocess_image(warped_rotated, rotated_rect)

        image = np.asarray(face_im)
        image = np.reshape(image, [1, image_size, image_size, 3])
        rating, grad = sess.run([hot_score, hot_grad], feed_dict={x: image})
        grad = np.reshape(grad, [-1])

    modelview, projection, viewport, indexs = decomposePose(mesh_orig, pose_orig, image_orig)
    results = {}
    results["rating"] = float(rating_orig)
    results["shape"] = shape_orig.tolist()
    results["texture"] = texture_orig.tolist()
    results["expression"] = expression_orig.tolist()

    results["new_rating"] = float(rating)
    results["new_shape"] = shape.tolist()
    results["new_texture"] = texture.tolist()

    results["modelview"] = modelview
    results["projection"] = projection
    results["viewport"] = viewport
    results["indexs"] = indexs

    return results, warpedim
