import csv
import os
import os.path
import sys

import cv2
import dlib
import numpy as np
import tensorflow as tf
from django.db.models import Count, Q

sys.path.append('./kaffe')
sys.path.append('./ResNet')
from processing.ResNet.ThreeDMM_shape import ResNet_101 as resnet101_shape

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ratemescraper.settings")
import django

django.setup()

from django.conf import settings
from rateme.models import Submission

tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

# inputlist = sys.argv[1]  # You can try './input.csv' or input your own file

# Global parameters
_tmpdir = './tmp/'  # save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet
print('> make dir')
if not os.path.exists(_tmpdir):
    os.makedirs(_tmpdir)
output_proc = 'output_preproc.csv'  # save intermediate image list
output_val = 'output_preproc_val.csv'  # save intermediate image list
factor = 0.25  # expand the given face bounding box to fit in the DCCNs
_alexNetSize = 227

# Get training image mean for Shape CNN
mean_image_shape = np.load('./processing/Shape_Model/3DMM_shape_mean.npy', fix_imports=True, encoding='bytes')  # 3 x 224 x 224
mean_image_shape = np.transpose(mean_image_shape, [1, 2, 0])  # 224 x 224 x 3, [0,255]

DLIB_SHAPEPREDICTOR_PATH = os.environ.get("DLIB_SHAPEPREDICTOR_PATH")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_SHAPEPREDICTOR_PATH)


def load_rateme_data():
    submissions = Submission.objects.filter(has_images=True, calculated_rating__isnull=False, gender='F') \
        .annotate(usable_comments_count=Count("comments", filter=Q(comments__rating__isnull=False))) \
        .filter(usable_comments_count__gte=10)

    af_data = []
    for submission in submissions:
        for image in submission.images.all():
            for face in image.face_processings.all():
                subject = face.id
                score = submission.calculated_rating

                # impath = os.path.join(settings.MEDIA_ROOT, face.roi.name)
                impath = "/test.jpg"

                row = (subject, impath, score)
                af_data.append(row)

    with open(output_proc, "w", newline='\n') as f:
        writer = csv.writer(f)

        # writer.writerow(("ID", "IM_PATH", "SCORE"))
        writer.writerows(af_data)

    return af_data


def extract_PSE_feats():
    # Prepare data
    load_rateme_data()

    # placeholders for the batches
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 1])

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

        with tf.variable_scope('shapeCNN_fc2'):
            dense = tf.layers.dense(inputs=fc1ls, units=256, activation=tf.nn.relu)
            # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        with tf.variable_scope('shapeCNN_fc3'):
            dense = tf.layers.dense(inputs=dense, units=32, activation=tf.nn.relu)
            # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

        with tf.variable_scope('shapeCNN_output'):
            # Logits Layer
            output = tf.layers.dense(inputs=dense, units=1)

    loss = tf.reduce_mean(tf.square(output - labels))

    # Configure the Training Op (for TRAIN mode)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_step())

    # Add ops to save and restore all the variables.
    init_op = tf.global_variables_initializer()
    saver_ini_shape_net = tf.train.Saver(
        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN')[:-6])

    all_saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        load_path = "./processing/Shape_Model/ini_ShapeTextureNet_model.ckpt"
        saver_ini_shape_net.restore(sess, load_path)

        # train for epochs
        for epoch in range(10):
            with open(output_proc, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')

                rows = []
                for i, row in enumerate(csvreader):
                    image_key = row[0]
                    image_file_path = row[1]
                    score = np.array([float(row[2]), ]).reshape([1, 1])

                    image = cv2.imread(image_file_path)  # BGR
                    image = np.asarray(image)

                    image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
                    op, preds_out, loss_out = sess.run([train_op, output, loss], feed_dict={x: image, labels: score})

                    print("epoch {}: sample {}, loss {}".format(epoch, i, loss_out))

                    if i > 0 and i % 100 == 0:
                        print("running validation:")
                        with open(output_val, 'r') as valfile:
                            valreader = csv.reader(valfile, delimiter=',')
                            avg_loss = []
                            for row in valreader:
                                image_key = row[0]
                                image_file_path = row[1]
                                score = np.array([float(row[2]), ]).reshape([1, 1])

                                image = cv2.imread(image_file_path, 1)  # BGR
                                image = np.asarray(image)

                                image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
                                preds_out, loss_out = sess.run([output, loss],
                                                               feed_dict={x: image, labels: score})

                                avg_loss.append(loss_out)
                            avg_loss = np.mean(avg_loss)
                            print("epoch {}: sample {}, VALIDATION {}".format(epoch, i, avg_loss))
                            all_saver.save(sess, './hotness_{}_{}.ckpt'.format(epoch, i))


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
        extract_PSE_feats()


if __name__ == '__main__':
    tf.app.run()
