import tensorflow as tf
import numpy as np

import sys
from utils.config import get

eye_side = get("TRAIN.EYE_SIDE")
pupil_side = get("TRAIN.PUPIL_SIDE")
diagonsis_size = len(get('DIAGNOSIS_MAP'))

class pupil_weight:
    # W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    # b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    # W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    # b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    # W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    # b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    # W4 = tf.Variable(tf.random_normal([pupil_side ** 2, 50], stddev = 1.0 / pupil_side))
    # b4 = tf.Variable(tf.constant(0.01, shape=[50]))

    filter_side = 3
    layer1_dim = 8
    layer2_dim = 16
    layer3_dim = 32

    W1 = tf.Variable(tf.truncated_normal([filter_side, filter_side, 3, layer1_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * 3)))
    b1 = tf.Variable(tf.constant(0.01, shape=[layer1_dim]))
    W2 = tf.Variable(tf.truncated_normal([filter_side, filter_side, layer1_dim, layer2_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * layer1_dim)))
    b2 = tf.Variable(tf.constant(0.01, shape=[layer2_dim]))
    W3 = tf.Variable(tf.truncated_normal([filter_side, filter_side, layer2_dim, layer3_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * layer2_dim)))
    b3 = tf.Variable(tf.constant(0.01, shape=[layer3_dim]))
    W4 = tf.Variable(tf.random_normal([pupil_side ** 2 / 64 * layer3_dim, 50], stddev = 0.1 / np.sqrt(pupil_side ** 2 / 64 * layer3_dim)))
    b4 = tf.Variable(tf.constant(0.01, shape=[50]))

class eye_weight:
    # W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    # b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    # W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    # b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    # W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    # b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    # W4 = tf.Variable(tf.random_normal([eye_side ** 2, 50], stddev = 1.0 / eye_side))
    # b4 = tf.Variable(tf.constant(0.01, shape=[50]))

    filter_side = 3
    layer1_dim = 8
    layer2_dim = 16
    layer3_dim = 32

    W1 = tf.Variable(tf.truncated_normal([filter_side, filter_side, 3, layer1_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * 3)))
    b1 = tf.Variable(tf.constant(0.01, shape=[layer1_dim]))
    W2 = tf.Variable(tf.truncated_normal([filter_side, filter_side, layer1_dim, layer2_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * layer1_dim)))
    b2 = tf.Variable(tf.constant(0.01, shape=[layer2_dim]))
    W3 = tf.Variable(tf.truncated_normal([filter_side, filter_side, layer2_dim, layer3_dim], stddev = 0.1 / np.sqrt(filter_side ** 2 * layer2_dim)))
    b3 = tf.Variable(tf.constant(0.01, shape=[layer3_dim]))
    W4 = tf.Variable(tf.random_normal([eye_side ** 2 / 64 * layer3_dim, 50], stddev = 0.1 / np.sqrt(eye_side ** 2 / 64 * layer3_dim)))
    b4 = tf.Variable(tf.constant(0.01, shape=[50]))

class final_weight:
    W = tf.Variable(tf.random_normal([100, diagonsis_size], stddev = 1.0 / np.sqrt(100)))
    # W = tf.Variable(tf.random_normal([50, diagonsis_size], stddev = 1.0 / np.sqrt(50)))
    b = tf.Variable(tf.constant(0.01, shape=[diagonsis_size]))

def pupil_cnn(pupil, keep_prob):
    conv1 = tf.nn.relu(tf.nn.conv2d(pupil, pupil_weight.W1, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, pupil_weight.W2, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, pupil_weight.W3, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b3)
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # conv1 = tf.nn.relu(tf.nn.conv2d(pupil, pupil_weight.W1, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b1)
    # conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # conv2 = tf.nn.relu(tf.nn.conv2d(conv1, pupil_weight.W2, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b2)
    # conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # conv3 = tf.nn.relu(tf.nn.conv2d(conv2, pupil_weight.W3, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b3)
    # conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    feature_vector = tf.reshape(conv3, [-1, pupil_side ** 2 / 64 * pupil_weight.layer3_dim])
    full_con = tf.nn.relu(tf.matmul(feature_vector, pupil_weight.W4) + pupil_weight.b4)

    return full_con

def eye_cnn(image, keep_prob):
    conv1 = tf.nn.relu(tf.nn.conv2d(image, eye_weight.W1, strides=[1, 2, 2, 1], padding='SAME') + eye_weight.b1)
    conv1 = tf.nn.dropout(conv1, keep_prob)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, eye_weight.W2, strides=[1, 2, 2, 1], padding='SAME') + eye_weight.b2)
    conv2 = tf.nn.dropout(conv2, keep_prob)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, eye_weight.W3, strides=[1, 2, 2, 1], padding='SAME') + eye_weight.b3)
    conv3 = tf.nn.dropout(conv3, keep_prob)
    feature_vector = tf.reshape(conv3, [-1, eye_side ** 2 / 64 * eye_weight.layer3_dim])
    full_con = tf.nn.relu(tf.matmul(feature_vector, eye_weight.W4) + eye_weight.b4)

    return full_con

def final_score(eye, pupil):
    combine = tf.concat([eye, pupil], 1)
    final_score = tf.matmul(combine, final_weight.W) + final_weight.b

    # final_score = tf.matmul(pupil, final_weight.W) + final_weight.b

    return final_score


def cnn():
    eye_size = 3 * eye_side ** 2
    pupil_size = 3 * pupil_side ** 2

    keep_prob = tf.placeholder(tf.float32)
    input_layer = tf.placeholder(tf.float32, shape=[None, eye_size + pupil_size])
    eye, pupil = tf.split(input_layer, [eye_size, pupil_size], 1)

    eye = tf.reshape(eye, [-1, eye_side, eye_side, 3])
    pupil = tf.reshape(pupil, [-1, pupil_side, pupil_side, 3])

    eye = eye_cnn(eye, keep_prob)
    pupil = pupil_cnn(pupil, keep_prob)

    pred_layer = final_score(eye, pupil)


    # regularizer = tf.nn.l2_loss(pupil_weight.W1) + tf.nn.l2_loss(pupil_weight.W2) + tf.nn.l2_loss(pupil_weight.W3)
    #     + tf.nn.l2_loss(eye_weight.W1) + tf.nn.l2_loss(eye_weight.W2) + tf.nn.l2_loss(eye_weight.W3)

    # pupil_size = 3 * pupil_side ** 2
    # input_layer = tf.placeholder(tf.float32, shape=[None, pupil_size])

    # pupil = tf.reshape(input_layer, [-1, pupil_side, pupil_side, 3])
    # pupil = pupil_cnn(pupil)

    # pred_layer = final_score(pupil)

    return input_layer, pred_layer, keep_prob




