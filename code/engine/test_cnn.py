import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets

if __name__ == '__main__':
    print('restoring model...')
    assert is_file_prefix(
        'TRAIN.CHECKPOINT'), "training checkpoint not found!"
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    input_layer, prediction_layer = cnn()  # fetch autoencoder layers
    saver = tf.train.Saver()  # prepare to restore weights
    saver.restore(sess, get('TRAIN.CHECKPOINT'))
    print('Yay! I restored weights from a saved model!')

    print('loading data...')
    data = read_data_sets(one_hot=True)
    Xs = data.test.images
    ys = data.test.labels

    print('computing reconstructions...')
    prediction = prediction_layer.eval(feed_dict={input_layer: Xs})

    print prediction
    print ys

    print average_precision_score(ys.ravel(), prediction.ravel())

