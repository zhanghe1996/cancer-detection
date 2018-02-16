import os
import pandas
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets

diagonsis_size = len(get('DIAGNOSIS_MAP'))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--set', dest='set', default='validation', type=str)

    args = parser.parse_args()

    return args

def load_test_data():
    eye_side = get("TRAIN.EYE_SIDE")
    pupil_side = get("TRAIN.PUPIL_SIDE")
    df = pandas.read_csv(get('DATA.TEST_CSV_PATH'))

    matrix = df.as_matrix()
    images = matrix[:, :(3 * eye_side ** 2 + 3 * pupil_side ** 2)]
    # images = matrix[:, :(3 * pupil_side ** 2)]
    meta_data = matrix[:, (3 * eye_side ** 2 + 3 * pupil_side ** 2):-1]
    labels = matrix[:, -1:]
    labels = np.reshape(labels, -1)

    new_labels = np.zeros((labels.shape[0], diagonsis_size), dtype = np.float32)
    for i in range(labels.shape[0]):
        new_labels[i][int(labels[i])] = 1
    labels = new_labels
    
    return images, labels

def getReverseDiagnosisMap():
    reverse_map = {}
    m = get('DIAGNOSIS_MAP')
    for key, value in m.iteritems():
        reverse_map[value] = key
    return reverse_map

def getAccuracy(ys, prediction):
    count = 0.0
    for index in range(prediction.shape[0]):
        if np.argmax(ys[index]) == np.argmax(prediction[index]):
            count += 1

    return count / ys.shape[0]

def getDetailedPerformance(ys, prediction):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for index in range(prediction.shape[0]):
        if np.argmax(prediction[index]) == 0:
            if np.argmax(ys[index]) == 0:
                tn += 1
            else:
                fn += 1
        else:
            if np.argmax(ys[index]) == 0:
                fp += 1
            else:
                tp += 1

    return tp, tn, fp, fn

def visualize(Xs, ys, prediction):
    rm = getReverseDiagnosisMap()

    eye_side = get('TRAIN.EYE_SIDE')
    pupil_side = get('TRAIN.PUPIL_SIDE')
    eye_size = eye_side * eye_side * 3
    pupil_size = pupil_side * pupil_side * 3
    for index in range(prediction.shape[0]):
        eye = Xs[index][:eye_size].reshape((eye_side, eye_side, 3))
        eye = eye.astype(np.float32)
        eye = (1 - eye) * 127.5
        pupil = Xs[index][eye_size : eye_size + pupil_size].reshape((pupil_side, pupil_side, 3))
        pupil = pupil.astype(np.float32)
        pupil = (1 - pupil) * 127.5
        plt.title('Ground Truth: %s, Prediction: %s' 
            %(
                rm[np.argmax(ys[index])], 
                rm[np.argmax(prediction[index])], 
            )
        )
        vetical_line = np.zeros((pupil_side, 1, 3))
        plt.imshow(np.hstack([eye, vetical_line, pupil]))
        plt.show()

    # pupil_side = get('TRAIN.PUPIL_SIDE')
    # pupil_size = pupil_side * pupil_side * 3

    # for index in range(prediction.shape[0]):
    #     pupil = Xs[index].reshape((pupil_side, pupil_side, 3))
    #     pupil = (1 - pupil) * 127.5
    #     gt = rm[np.argmax(ys[index])]
    #     p = rm[np.argmax(prediction[index])]
    #     plt.title('Ground Truth: %s, Prediction: %s' %(gt, p))
    #     vetical_line = np.zeros((pupil_side, 1, 3))
    #     plt.imshow(np.hstack([pupil]))
    #     plt.show()

def test_on_checkpoints(iters, Xs, ys, if_visualize):
    for iter in iters:
        print 'iteration ' + str(iter) + ': '

        sess = tf.InteractiveSession()  # start talking to tensorflow backend
        saver = tf.train.Saver()  # prepare to restore weights
        saver.restore(sess, os.path.join('..', 'checkpoints', 'iter_' + str(iter), 'cnn.ckpt'))
        # saver.restore(sess, get('TRAIN.CHECKPOINT'))
        
        print('computing reconstructions...')
        prediction = prediction_layer.eval(feed_dict={input_layer: Xs, keep_prob: 1.0})

        # print 'Ground True:'
        # print ys
        # print 'Prediction:'
        # print prediction

        print 'Total precision: %f' %average_precision_score(ys.ravel(), prediction.ravel())
        print 'Total accuracy: %f' %getAccuracy(ys, prediction)

        tp, tn, fp, fn = getDetailedPerformance(ys, prediction)
        print 'True Positive: %d' %tp
        print 'True Negative: %d' %tn
        print 'False Positive: %d' %fp
        print 'False Negative: %d' %fn

        if if_visualize:
            visualize(Xs, ys, prediction)

if __name__ == '__main__':
    args = parse_args()
    print('restoring model...')
    input_layer, prediction_layer, keep_prob = cnn()  # fetch autoencoder layers


    print('loading data...')
    data = read_data_sets(one_hot=True, balance_classes=False)

    if args.set == 'train':
        Xs = data.train.images
        ys = data.train.labels
    elif args.set == 'validation':
        Xs = data.validation.images
        ys = data.validation.labels
    else:
        Xs, ys = load_test_data()

    iters = [200, 400, 600, 800, 1000]
    test_on_checkpoints(iters, Xs, ys, False)


    # iter = 200
    # while iter < get('TRAIN.NB_STEPS'):
    #     print 'iteration ' + str(iter) + ': '

    #     saver.restore(sess, os.path.join('..', 'checkpoints', 'iter_' + str(iter), 'cnn.ckpt'))
    #     # saver.restore(sess, get('TRAIN.CHECKPOINT'))
        
    #     print('Yay! I restored weights from a saved model!')
    #     print('loading data...')
    #     data = read_data_sets(one_hot=True, balance_classes=False)

    #     if args.test:
    #         Xs, ys = load_test_data()
    #     else: 
    #         if args.set == 'train':
    #             Xs = data.train.images
    #             ys = data.train.labels
    #         elif args.set == 'validation':
    #             Xs = data.validation.images
    #             ys = data.validation.labels
    #         else:
    #             Xs = data.test.images
    #             ys = data.test.labels

    #     print('computing reconstructions...')
    #     prediction = prediction_layer.eval(feed_dict={input_layer: Xs})

    #     # print 'Ground True:'
    #     # print ys
    #     # print 'Prediction:'
    #     # print prediction

    #     print 'Total precision: %f' %average_precision_score(ys.ravel(), prediction.ravel())
    #     print 'Total accuracy: %f' %getAccuracy(ys, prediction)

    #     tp, tn, fp, fn = getDetailedPerformance(ys, prediction)
    #     print 'True Positive: %d' %tp
    #     print 'True Negative: %d' %tn
    #     print 'False Positive: %d' %fp
    #     print 'False Negative: %d' %fn

    #     iter += 200



