import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets

def visualize(Xs, ys, prediction):
    image_side = get('TRAIN.IMAGE_SIDE')
    pupil_side = get('TRAIN.PUPIL_SIDE')
    image_size = image_side * image_side * 3
    pupil_size = pupil_side * pupil_side * 3
    for index in range(prediction.shape[0]):
        # image = Xs[index][:image_size].reshape((image_side, image_side, 3))
        # print (1 - image) * 127.5
        # plt.imshow((1 - image) * 127.5)
        # plt.show()

        left_pupil = Xs[index][image_size : image_size + pupil_size].reshape((pupil_side, pupil_side, 3))
        plt.imshow((1 - left_pupil) * 127.5)
        plt.title('Left Pupil\n Ground Truth: %f, Prediction: %f' %(ys[index][1], prediction[index][1]))
        plt.show()

        right_pupil = Xs[index][image_size + pupil_size:].reshape((pupil_side, pupil_side, 3))
        right_pupil[:][:][0], right_pupil[:][:][2] = right_pupil[:][:][2], right_pupil[:][:][0]
        plt.imshow((1 - right_pupil) * 127.5)
        plt.title('Right Pupil\n Ground Truth: %f, Prediction: %f' %(ys[index][3], prediction[index][3]))
        plt.show()

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

    print 'Total precision: %f' %average_precision_score(ys.ravel(), prediction.ravel())

    visualize(Xs, ys, prediction)



