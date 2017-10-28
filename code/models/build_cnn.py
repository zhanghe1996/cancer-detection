import tensorflow as tf
import numpy as np

import sys
from utils.config import get

def average_pooling(x, in_length=32, scale=2):
    as_image = tf.reshape(x, [-1, in_length, in_length, 3])
    pooled = tf.nn.avg_pool(as_image, ksize=[1, scale, scale, 1], strides=[1, scale, scale, 1], padding='SAME')
    as_vector = tf.reshape(pooled, [-1, (in_length//scale)**2]) 
    return as_vector

def pupil_cnn(pupil):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    W4 = tf.Variable(tf.random_normal([2 * 2 * 64, 50], stddev = 1.0 / 16))
    b4 = tf.Variable(tf.constant(0.01, shape=[50]))
    W5 = tf.Variable(tf.random_normal([50, 1], stddev = 1.0 / np.sqrt(50)))
    b5 = tf.Variable(tf.constant(0.01, shape=[1]))

    conv1 = tf.nn.relu(tf.nn.conv2d(pupil, W1, strides=[1, 2, 2, 1], padding='SAME') + b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides=[1, 2, 2, 1], padding='SAME') + b3)
    feature_vector = tf.reshape(conv3, [-1, 2 * 2 * 64])
    full_con = tf.nn.relu(tf.matmul(feature_vector, W4) + b4)
    result = tf.matmul(full_con, W5) + b5

    return result

def image_cnn(image):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 16], stddev = 0.1 / np.sqrt(75)))
    b1 = tf.Variable(tf.constant(0.01, shape=[16]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev = 0.1 / 20))
    b2 = tf.Variable(tf.constant(0.01, shape=[32]))
    W3 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev = 0.1 / np.sqrt(800)))
    b3 = tf.Variable(tf.constant(0.01, shape=[64]))
    W4 = tf.Variable(tf.random_normal([4 * 4 * 64, 50], stddev = 1.0 / 16))
    b4 = tf.Variable(tf.constant(0.01, shape=[50]))
    W5 = tf.Variable(tf.random_normal([50, 1], stddev = 1.0 / np.sqrt(50)))
    b5 = tf.Variable(tf.constant(0.01, shape=[1]))

    conv1 = tf.nn.relu(tf.nn.conv2d(image, W1, strides=[1, 2, 2, 1], padding='SAME') + b1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides=[1, 2, 2, 1], padding='SAME') + b3)
    feature_vector = tf.reshape(conv3, [-1, 4 * 4 * 64])
    full_con = tf.nn.relu(tf.matmul(feature_vector, W4) + b4)
    result = tf.matmul(full_con, W5) + b5

    return result

def final_score(image, pupil):
    combine = tf.concat([image, pupil], 1)
    W = tf.Variable(tf.random_normal([2, 2], stddev = 1.0 / 2))
    b = tf.Variable(tf.constant(0.01, shape=[2]))

    final_score = tf.nn.relu(tf.matmul(combine, W) + b)
    final_score = tf.nn.softmax(final_score)

    return final_score


def cnn():
    image_side = get("TRAIN.IMAGE_SIDE")
    pupil_side = get("TRAIN.PUPIL_SIDE")
    image_size = 3 * image_side ** 2
    pupil_size = 3 * pupil_side ** 2

    input_layer = tf.placeholder(tf.float32, shape=[None, image_size + pupil_size * 2])
    image, pupil_left, pupil_right = tf.split(input_layer, [image_size, pupil_size, pupil_size], 1)

    image = tf.reshape(image, [-1, image_side, image_side, 3])
    pupil_left = tf.reshape(pupil_left, [-1, pupil_side, pupil_side, 3])
    pupil_right = tf.reshape(pupil_right, [-1, pupil_side, pupil_side, 3])

    image = image_cnn(image)
    pupil_left = pupil_cnn(pupil_left)
    pupil_right = pupil_cnn(pupil_right)

    score_left = final_score(image, pupil_left)
    score_right = final_score(image, pupil_right)

    pred_layer = tf.concat([score_left, score_right], 1)

    return input_layer, pred_layer




