import os
import pandas
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import xml.dom.minidom as minidom
from sklearn.metrics import average_precision_score

from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets
from data_scripts.load_data import getEyeIndex, get_data_from_tag

diagonsis_size = len(get('DIAGNOSIS_MAP'))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--set', dest='set', default='validation', type=str)

    args = parser.parse_args()

    return args

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

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
        saver.restore(sess, os.path.join('..', 'checkpoints', 'iter_' + str(iter), 'cnn.ckpt')) # load weights from checkpoints
        # saver.restore(sess, get('TRAIN.CHECKPOINT'))
        
        print('computing reconstructions...')
        prediction = prediction_layer.eval(feed_dict={input_layer: Xs, keep_prob: 1.0}) # obtian predictions

        # compute softmax
        prediction_softmax = []
        for p in prediction:
            prediction_softmax.append(softmax(p))

        prediction_softmax = np.asarray(prediction_softmax)

        # print 'Ground True:'
        # print ys
        # print 'Prediction:'
        # print prediction_softmax

        print 'Total precision: %f' %average_precision_score(ys.ravel(), prediction_softmax.ravel())
        print 'Total accuracy: %f' %getAccuracy(ys, prediction_softmax)

        tp, tn, fp, fn = getDetailedPerformance(ys, prediction_softmax)
        print 'True Positive: %d' %tp
        print 'True Negative: %d' %tn
        print 'False Positive: %d' %fp
        print 'False Negative: %d' %fn

        if if_visualize:
            visualize(Xs, ys, prediction_softmax)


def check_single_image(image_name, iter):
    eye_side = get("TRAIN.EYE_SIDE")
    pupil_side = get("TRAIN.PUPIL_SIDE")
    im_file = os.path.join(get("DATA.DATA_PATH"), 'Images', image_name + '.jpeg')
    eye_file = os.path.join(get("DATA.DATA_PATH"), 'Annotations', 'eye', image_name + '_eye.npy')
    pupil_file = os.path.join(get("DATA.DATA_PATH"), 'Annotations', 'pupil', image_name + '.xml')

    im = Image.open(im_file)

    with open(pupil_file) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    if num_objs != 2:
        print "Error: the number of pupil is not two"
        return

    pupil_boxes = np.zeros((num_objs, 4), dtype = np.uint16)


    # load pupil position in xml file
    for ix, obj in enumerate(objs):
        pupil_boxes[ix][0] = int(get_data_from_tag(obj, 'xmin'))
        pupil_boxes[ix][1] = int(get_data_from_tag(obj, 'ymin'))
        pupil_boxes[ix][2] = int(get_data_from_tag(obj, 'xmax'))
        pupil_boxes[ix][3] = int(get_data_from_tag(obj, 'ymax'))

    # exchange boxes when pupil_boxes[0] is actually the right pupil
    if pupil_boxes[0][0] < pupil_boxes[1][0]:
        pupil_boxes[[0, 1]] = pupil_boxes[[1, 0]]

    eye_boxes = np.load(eye_file)

    Xs = []

    # generate new lines of data 
    for i in range(2):
        pupil = im.crop(pupil_boxes[i])
        pupil = pupil.resize((pupil_side, pupil_side), Image.ANTIALIAS)
        pupil_pix = np.array(pupil, dtype = np.float32)

        pupil_pix = np.reshape(pupil_pix, (-1))
        pupil_pix = pupil_pix / 127.5 - 1

        eye_index = getEyeIndex(eye_boxes, pupil_boxes[i])

        if eye_index == -1:
            print "Cannot find corresponding eye box"
            print pupil_boxes[i]
            print np.array(eye_boxes, dtype=np.int32)
            continue

        eye = im.crop(eye_boxes[eye_index][:-1])
        eye = eye.resize((eye_side, eye_side), Image.ANTIALIAS)
        eye_pix = np.array(eye, dtype = np.float32)

        eye_pix = np.reshape(eye_pix, (-1))
        eye_pix = eye_pix / 127.5 - 1

        feature = np.append(eye_pix, pupil_pix)

        feature = feature.tolist()

        assert(len(feature) == 3 * eye_side ** 2 + 3 * pupil_side ** 2)

        Xs.append(feature)


    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    saver = tf.train.Saver()  # prepare to restore weights
    saver.restore(sess, os.path.join('..', 'checkpoints', 'iter_' + str(iter), 'cnn.ckpt')) # load weights from checkpoints

    print('computing reconstructions...')
    prediction = prediction_layer.eval(feed_dict={input_layer: Xs, keep_prob: 1.0}) # obtian predictions

    # compute softmax
    prediction_softmax = []
    for p in prediction:
        prediction_softmax.append(softmax(p).tolist())

    print 'Prediction:'
    print prediction_softmax

if __name__ == '__main__':
	
    args = parse_args()
    print('restoring model...')
    input_layer, prediction_layer, keep_prob = cnn()  # fetch autoencoder layers

    check_single_image("2b51c237da310ac10f47f9841c49b05b_FORWARD_GAZE", 800)

    # print('loading data...')
    # data = read_data_sets(one_hot=True, balance_classes=False)

    # # get corresponding dataset according to flag
    # if args.set == 'train':
    #     Xs = data.train.images
    #     ys = data.train.labels
    # elif args.set == 'validation':
    #     Xs = data.validation.images
    #     ys = data.validation.labels
    # else:
    #     Xs, ys = load_test_data()

    # iters = [200, 400, 600, 800, 1000]
    # test_on_checkpoints(iters, Xs, ys, False)



