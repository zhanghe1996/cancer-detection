import os
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base

from utils.config import get
from data_scripts.pupils2017 import PUPILS2017


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_set(data, split_name, one_hot, balance_classes):
    images, labels = data.preprocessed_data(split_name,
                                            one_hot=one_hot,
                                            balance_classes=balance_classes)
    return images, labels


def read_data_sets(one_hot=False,
                   balance_classes=False,
                   dtype=dtypes.float32,
                   reshape=True):
    data = PUPILS2017()

    train_images, train_labels = data.preprocessed_data('train',
                                                        one_hot=one_hot,
                                                        balance_classes=balance_classes)
    val_images, val_labels = data.preprocessed_data('val',
                                                    one_hot=one_hot,
                                                    balance_classes=balance_classes)
    test_images, test_labels = data.preprocessed_data('test',
                                                      one_hot=one_hot,
                                                      balance_classes=balance_classes)

    train = DataSet(train_images,
                    train_labels,
                    dtype=dtype,
                    reshape=reshape)
    validation = DataSet(val_images,
                         val_labels,
                         dtype=dtype,
                         reshape=reshape)
    test = DataSet(test_images,
                   test_labels,
                   dtype=dtype,
                   reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)
