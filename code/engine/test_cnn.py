import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--set', dest='set', default='test', type=str)

    args = parser.parse_args()

    return args

def visualize(Xs, ys, prediction):
    image_side = get('TRAIN.IMAGE_SIDE')
    pupil_side = get('TRAIN.PUPIL_SIDE')
    image_size = image_side * image_side * 3
    pupil_size = pupil_side * pupil_side * 3
    for index in range(prediction.shape[0]):
        left_pupil = Xs[index][image_size : image_size + pupil_size].reshape((pupil_side, pupil_side, 3))
        left_pupil = (1 - left_pupil) * 127.5
        right_pupil = Xs[index][image_size + pupil_size:].reshape((pupil_side, pupil_side, 3))
        right_pupil = (1 - right_pupil) * 127.5
        plt.title('Left Pupil: {Ground Truth: %f, Prediction: %f}\n Right Pupil: {Ground Truth: %f, Prediction: %f}' 
            %(ys[index][1], prediction[index][1], ys[index][3], prediction[index][3]))
        vetical_line = np.zeros((pupil_side, 1, 3))
        plt.imshow(np.hstack([left_pupil, vetical_line, right_pupil]))
        plt.show()



if __name__ == '__main__':
    args = parse_args()
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

    if args.set == 'train':
        Xs = data.train.images
        ys = data.train.labels
    elif args.set == 'validation':
        Xs = data.validation.images
        ys = data.validation.labels
    else:
        Xs = data.test.images
        ys = data.test.labels

    print('computing reconstructions...')
    prediction = prediction_layer.eval(feed_dict={input_layer: Xs})

    print 'Ground True:'
    print ys
    print 'Prediction:'
    print prediction

    print 'Total precision: %f' %average_precision_score(ys.ravel(), prediction.ravel())

    visualize(Xs, ys, prediction)



