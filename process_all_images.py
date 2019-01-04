import json
import math
import os
import os.path
import sys
import traceback

import cv2
import dlib
import numpy as np
import tensorflow as tf
from django.db.models import Count, F

os.environ.setdefault("DLIB_SHAPEPREDICTOR_PATH", "./processing/shape_predictor_68_face_landmarks.dat")
os.environ.setdefault("EOS_DATA_PATH", "./processing/share")

from processing.Beautifier.face3D.faceFeatures3D import BFM_FACEFITTING
from processing.Beautifier.faceFeatures import getLandmarks, LEFT_EYE, RIGHT_EYE

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings
from rateme.models import SubmissionImage, FaceProcessing

sys.path.append('./processing/kaffe')
sys.path.append('./processing/ResNet')
from processing.ResNet.ThreeDMM_shape import ResNet_101 as resnet101_shape
from processing.ResNet.ThreeDMM_expr import ResNet_101 as resnet101_expr
from processing.im_utils import image_resize, rotate_bound, ensure_max_image_size

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

# Global parameters
_tmpdir = './tmp/'  # save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet
output_proc = 'output_preproc.csv'  # save intermediate image list
output_val = 'output_preproc_val.csv'  # save intermediate image list
factor = 0.25  # expand the given face bounding box to fit in the DCCNs
_alexNetSize = 227

# Get training image mean for Shape CNN
mean_image_shape = np.load('./processing/Shape_Model/3DMM_shape_mean.npy', fix_imports=True,
                           encoding='bytes')  # 3 x 224 x 224
mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]

DLIB_SHAPEPREDICTOR_PATH = os.environ.get("DLIB_SHAPEPREDICTOR_PATH")
MEDIA_ROOT = settings.MEDIA_ROOT
MAX_ALIGN_ATTEMPTS = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPEPREDICTOR_PATH)


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


def extract_facial_features():
    # set up the network
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, None, None, 3])

    ###################
    # Shape CNN
    ###################
    x2 = tf.image.resize_bilinear(x, tf.constant([224, 224], dtype=tf.int32))
    x2 = tf.cast(x2, 'float32')
    x2 = tf.reshape(x2, [FLAGS.batch_size, 224, 224, 3])

    # Image normalization
    mean = tf.reshape(mean_image_shape, [1, 224, 224, 3])
    mean = tf.cast(mean, 'float32')
    x2 = x2 - mean

    with tf.variable_scope('shapeCNN'):
        net_shape = resnet101_shape({'input': x2}, trainable=False)
        pool5 = net_shape.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])

        npzfile = np.load('./processing/ResNet/ShapeNet_fc_weights.npz', fix_imports=True, encoding='bytes')
        ini_weights_shape = npzfile['ini_weights_shape']
        ini_biases_shape = npzfile['ini_biases_shape']
        with tf.variable_scope('shapeCNN_fc1'):
            fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048, -1]), trainable=False, name='weights')
            fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=False, name='biases')
            fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)

    ###################
    # Expression CNN
    ###################
    with tf.variable_scope('exprCNN'):
        net_expr = resnet101_expr({'input': x2}, trainable=True)
        pool5 = net_expr.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size, -1])

        npzfile = np.load('./processing/ResNet/ExpNet_fc_weights.npz', fix_imports=True, encoding='bytes')
        ini_weights_expr = npzfile['ini_weights_expr']
        ini_biases_expr = npzfile['ini_biases_expr']
        with tf.variable_scope('exprCNN_fc1'):
            fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048, 29]), trainable=True, name='weights')
            fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
            fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)

    # Add ops to save and restore all the variables.
    init_op = tf.global_variables_initializer()
    all_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
    saver_ini_shape_net = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
    saver_ini_expr_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        load_path = "./processing/Shape_Model/ini_ShapeTextureNet_model.ckpt"
        saver_ini_shape_net.restore(sess, load_path)

        # load our expression net model
        load_path = "./processing/Expression_Model/ini_exprNet_model.ckpt"
        saver_ini_expr_net.restore(sess, load_path)

        images_to_process = SubmissionImage.objects.all() \
            .annotate(face_processing_count=Count("face_processings")) \
            .filter(face_processing_count__lt=F("face_count"))
        images_to_process_len = len(images_to_process)

        for sub_i, submissionimage in enumerate(images_to_process):
            try:
                relative_im_path = submissionimage.image.name
                im_path = os.path.join(MEDIA_ROOT, relative_im_path)

                print("{}/{}: {}".format(sub_i, images_to_process_len, im_path))

                # load the image and resize it to something reasonable
                image_orig_full = cv2.imread(im_path)
                image_orig = ensure_max_image_size(image_orig_full)
                image_resized = image_resize(image_orig_full)

                h_scale = image_orig.shape[0] / image_resized.shape[0]
                w_scale = image_orig.shape[1] / image_resized.shape[1]

                # get the location of the face in the image
                rects = detector(image_resized, 1)
                for face_id, rect in enumerate(rects):
                    try:
                        rect_resized = rect
                        rect_orig = dlib.rectangle(
                            left=int(math.floor(rect_resized.left() * w_scale)),
                            top=int(math.floor(rect_resized.top() * h_scale)),
                            right=int(math.ceil(rect_resized.right() * w_scale)),
                            bottom=int(math.ceil(rect_resized.bottom() * h_scale)))

                        # align the image so that the face is vertical
                        image_rotated, M, invM, rotated_rect = align_face_vertical(image_orig, rect_orig)
                        face_im, rect_rotated = preprocess_image(image_rotated, rect=None, guess_rect=rotated_rect)

                        print("\trunning shape and expression networks")
                        image = np.asarray(face_im)
                        image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
                        Shape_Color_orig, Expr = sess.run([fc1ls, fc1le], feed_dict={x: image})
                        Shape_Color_orig = np.reshape(Shape_Color_orig, [-1])
                        expr = np.reshape(Expr, [-1])

                        shape = Shape_Color_orig[0:99]
                        color = Shape_Color_orig[99:]

                        print("\tconverting to eos structures")
                        landmarks_rot = getLandmarks(image_rotated, rect_rotated)
                        landmarks_cor = np.array(
                            [(np.matrix(invM) * np.append(p, 1)[:, np.newaxis]) for p in landmarks_rot],
                            dtype=np.int).squeeze()
                        # landmarks_cor_large = np.array([[p[0] * w_scale, p[1] * h_scale] for p in landmarks_cor], dtype=np.int)

                        mesh_orig = BFM_FACEFITTING.getMeshFromShapeCeoffs(shape, expr, color)
                        pose_orig = BFM_FACEFITTING.getPoseFromMesh(landmarks_cor, mesh_orig, image_orig)

                        print("\textracting mesh texture")
                        max_side = np.max(np.max(landmarks_cor, axis=0) - np.min(landmarks_cor, axis=0)) * 2
                        texture = BFM_FACEFITTING.getTextureFromMesh(image_orig, mesh_orig, pose_orig,
                                                                     texture_size=int(max_side))

                        # save the shape, color, texture, expression to the database
                        print("\tsaving to db")
                        shape_str = json.dumps(shape.tolist())
                        color_str = json.dumps(color.tolist())
                        expr_str = json.dumps(expr.tolist())
                        pitch, yaw, roll = pose_orig.get_rotation_euler_angles()

                        file_type = os.path.splitext(im_path)[1]
                        relative_texture_path = relative_im_path.replace(file_type,
                                                                         "_face{}_texture.jpg".format(face_id))
                        texture_path = os.path.join(MEDIA_ROOT, relative_texture_path)
                        cv2.imwrite(texture_path, texture[:, :, :3])

                        imgprocessing = FaceProcessing(image=submissionimage,
                                                       texture=relative_texture_path,
                                                       shape_coefficients=shape_str,
                                                       color_coefficients=color_str,
                                                       expression_coefficients=expr_str,
                                                       pitch=pitch,
                                                       yaw=yaw,
                                                       roll=roll)
                        imgprocessing.save()
                    except Exception as e:
                        traceback.print_exc()
            except Exception as e:
                traceback.print_exc()


def main(_):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    # print dev
    with tf.device(dev):
        extract_facial_features()


if __name__ == '__main__':
    # extract_facial_features()
    tf.app.run()
