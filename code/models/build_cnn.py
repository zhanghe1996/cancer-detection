import tensorflow as tf
import numpy as np

import sys
from utils.config import get

diagonsis_size = len(get('DIAGNOSIS_MAP'))

class pupil_weight:
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    W4 = tf.Variable(tf.random_normal([4 * 4 * 64, 100], stddev = 1.0 / 32))
    b4 = tf.Variable(tf.constant(0.01, shape=[100]))

    # W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev = 0.1 / np.sqrt(75)))
    # b1 = tf.Variable(tf.constant(0.01, shape=[32]))
    # W2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / 20))
    # b2 = tf.Variable(tf.constant(0.01, shape=[64]))
    # W3 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev = 0.1 / np.sqrt(800)))
    # b3 = tf.Variable(tf.constant(0.01, shape=[128]))
    # W4 = tf.Variable(tf.random_normal([4 * 4 * 128, 100], stddev = 1.0 / 32))
    # b4 = tf.Variable(tf.constant(0.01, shape=[100]))

class image_weight:
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    W4 = tf.Variable(tf.random_normal([4 * 4 * 64, 10], stddev = 1.0 / 32))
    b4 = tf.Variable(tf.constant(0.01, shape=[10]))

class final_weight:
    # W = tf.Variable(tf.random_normal([110, diagonsis_size], stddev = 1.0 / np.sqrt(110)))
    # b = tf.Variable(tf.constant(0.01, shape=[diagonsis_size]))
    W = tf.Variable(tf.random_normal([100, diagonsis_size], stddev = 1.0 / np.sqrt(100)))
    b = tf.Variable(tf.constant(0.01, shape=[diagonsis_size]))

def pupil_cnn(pupil):
    conv1 = tf.nn.relu(tf.nn.conv2d(pupil, pupil_weight.W1, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, pupil_weight.W2, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, pupil_weight.W3, strides=[1, 2, 2, 1], padding='SAME') + pupil_weight.b3)

    # conv1 = tf.nn.relu(tf.nn.conv2d(pupil, pupil_weight.W1, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b1)
    # conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # conv2 = tf.nn.relu(tf.nn.conv2d(conv1, pupil_weight.W2, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b2)
    # conv2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # conv3 = tf.nn.relu(tf.nn.conv2d(conv2, pupil_weight.W3, strides=[1, 1, 1, 1], padding='SAME') + pupil_weight.b3)
    # conv3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    
    feature_vector = tf.reshape(conv3, [-1, 4 * 4 * 64])
    # feature_vector = tf.reshape(conv3, [-1, 4 * 4 * 128])
    full_con = tf.nn.relu(tf.matmul(feature_vector, pupil_weight.W4) + pupil_weight.b4)

    return full_con

def image_cnn(image):
    conv1 = tf.nn.relu(tf.nn.conv2d(image, image_weight.W1, strides=[1, 2, 2, 1], padding='SAME') + image_weight.b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, image_weight.W2, strides=[1, 2, 2, 1], padding='SAME') + image_weight.b2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, image_weight.W3, strides=[1, 2, 2, 1], padding='SAME') + image_weight.b3)
    feature_vector = tf.reshape(conv3, [-1, 4 * 4 * 64])
    full_con = tf.nn.relu(tf.matmul(feature_vector, image_weight.W4) + image_weight.b4)

    return full_con

def final_score(pupil):
    # combine = tf.concat([image, pupil], 1)
    
    # final_score = tf.matmul(combine, final_weight.W) + final_weight.b
    # final_score = tf.nn.softmax(final_score)

    final_score = tf.matmul(pupil, final_weight.W) + final_weight.b
    # final_score = tf.nn.softmax(final_score)

    return final_score


def cnn():
    # image_side = get("TRAIN.IMAGE_SIDE")
    # pupil_side = get("TRAIN.PUPIL_SIDE")
    # image_size = 3 * image_side ** 2
    # pupil_size = 3 * pupil_side ** 2

    # input_layer = tf.placeholder(tf.float32, shape=[None, image_size + pupil_size * 2])
    # image, pupil_left, pupil_right = tf.split(input_layer, [image_size, pupil_size, pupil_size], 1)

    # image = tf.reshape(image, [-1, image_side, image_side, 3])
    # pupil_left = tf.reshape(pupil_left, [-1, pupil_side, pupil_side, 3])
    # pupil_right = tf.reshape(pupil_right, [-1, pupil_side, pupil_side, 3])

    # image = image_cnn(image)
    # pupil_left = pupil_cnn(pupil_left)
    # pupil_right = pupil_cnn(pupil_right)

    # score_left = final_score(image, pupil_left)
    # score_right = final_score(image, pupil_right)

    # pred_layer = tf.concat([score_left, score_right], 1)

    pupil_side = get("TRAIN.PUPIL_SIDE")
    pupil_size = 3 * pupil_side ** 2
    input_layer = tf.placeholder(tf.float32, shape=[None, pupil_size])

    pupil = tf.reshape(input_layer, [-1, pupil_side, pupil_side, 3])
    pupil = pupil_cnn(pupil)

    pred_layer = final_score(pupil)

    return input_layer, pred_layer




