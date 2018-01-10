import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets

gaze_size = len(get('DATA.GAZES'))
diagonsis_size = len(get('DIAGNOSIS_MAP'))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--set', dest='set', default='validation', type=str)

    args = parser.parse_args()

    return args

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
    vetical_line = np.zeros((pupil_side, 1, 3))

    for i in range(prediction.shape[0]):
        for j in range(gaze_size):
            eye = Xs[i][(eye_size + pupil_size) * j : (eye_size + pupil_size) * j + eye_size].reshape((eye_side, eye_side, 3))
            eye = (1 - eye) * 127.5
            pupil = Xs[i][(eye_size + pupil_size) * j + eye_size : (eye_size + pupil_size) * (j + 1)].reshape((pupil_side, pupil_side, 3))
            pupil = (1 - pupil) * 127.5
            plt.title('Ground Truth: %s, Prediction: %s' 
                %(
                    rm[np.argmax(ys[i])], 
                    rm[np.argmax(prediction[i])], 
                )
            )
            if j == 0: 
                im = np.hstack([eye, vetical_line, pupil])
            else:
                im = np.vstack((im, np.hstack([eye, vetical_line, pupil])))

        plt.imshow(im)
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
    data = read_data_sets(one_hot=True, balance_classes=False)

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
    print 'Total accuracy: %f' %getAccuracy(ys, prediction)

    tp, tn, fp, fn = getDetailedPerformance(ys, prediction)
    print 'True Positive: %d' %tp
    print 'True Negative: %d' %tn
    print 'False Positive: %d' %fp
    print 'False Negative: %d' %fn

    visualize(Xs, ys, prediction)



