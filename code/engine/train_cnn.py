import os
import tensorflow as tf
from models.build_cnn import cnn
from utils.config import get, is_file_prefix
from data_scripts.pupils2017_dataset import read_data_sets


def get_weights(saver, sess):
    ''' load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CHECKPOINT'))
        print('Yay! I restored weights from a saved model!')
    else:
        print('OK, I did not find a saved model, so I will start training from scratch!')


def report_training_progress(batch_index, input_layer, keep_prob, loss_func, data):
    ''' Update user on training progress '''
    if batch_index % 5:
        return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 50:
        return

    error = loss_func.eval(feed_dict={input_layer: data.train.images, 
                        true_labels: data.train.labels, keep_prob: 1.0})
    # acc = accuracy.eval(feed_dict={input_layer: data.train.images, 
                        # true_labels: data.train.labels})
    print('\n \t training cross_entropy is about %f' % error)
    # print(' \t training accuracy is about %f' % acc)

    error = loss_func.eval(feed_dict={input_layer: data.validation.images, 
                           true_labels: data.validation.labels, keep_prob: 1.0})
    # acc = accuracy.eval(feed_dict={input_layer: data.validation.images, 
                        # true_labels: data.validation.labels})
    print(' \t validation cross_entropy is about %f' % error)
    # print(' \t validation accuracy is about %f' % acc)


def train_cnn(input_layer, prediction_layer, keep_prob, loss_func, optimizer, data, sess):
    ''' Train CNN '''
    try:
        for batch_index in range(get('TRAIN.NB_STEPS')):
            report_training_progress(batch_index, input_layer, keep_prob, loss_func, data) 
            batch_images, batch_labels = data.train.next_batch(get('TRAIN.BATCH_SIZE'))
            optimizer.run(feed_dict={input_layer: batch_images, true_labels: batch_labels, keep_prob: get('TRAIN.DROP_OUR_KEEP_PROB')})

            if batch_index != 0 and batch_index % 200 == 0:
                path = os.path.join('..', 'checkpoints', 'iter_' + str(batch_index))
                if not os.path.exists(path):
                    os.makedirs(path)
                saver.save(sess, os.path.join(path, 'cnn.ckpt'))

    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.')


if __name__ == '__main__':
    print('building model...')
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    input_layer, prediction_layer, keep_prob = cnn()  # fetch model layers
    # true_labels = tf.placeholder(tf.float32, shape=[None, len(get('DIAGNOSIS_MAP')) * 2])
    true_labels = tf.placeholder(tf.int32, shape=[None])
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=true_labels, logits=prediction_layer)) # define the loss function
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_labels, logits=prediction_layer))
    correct_prediction = tf.equal(tf.argmax(prediction_layer, 1), tf.argmax(true_labels, 1)) # define the correct predictions calculation
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # calculate accuracy
    optimizer = tf.train.AdamOptimizer(get('TRAIN.LEARNING_RATE')).minimize(cross_entropy + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0.01) # define the training step
    sess.run(tf.global_variables_initializer())  # initialize some globals
    saver = tf.train.Saver()  # prepare to save model
    # load model weights if they were saved previously
    get_weights(saver, sess)

    print('loading data...')
    data = read_data_sets(one_hot=False, balance_classes=False)

    print('training...')
    train_cnn(input_layer, prediction_layer, keep_prob, cross_entropy, optimizer, data, sess)

    print('saving trained model...\n')
    # saver.save(sess, get('TRAIN.CHECKPOINT'))
