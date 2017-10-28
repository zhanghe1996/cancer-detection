import numpy as np
from scipy.misc import imresize
from sklearn.utils import resample
import pandas

from utils.config import get, print_if_verbose

def remove_images(images, labels):
    removed = []
    for i in range(images.shape[0]):
        if np.linalg.norm(images[i] - np.mean(images[i])) < 1:
            removed.append(i)

    images = np.delete(images, removed, axis = 0)
    labels = np.delete(labels, removed, axis = 0)

    return len(removed), images, labels


class PUPILS2017:
    filename = ''
    data_stored = False
    train_images = np.zeros(0)
    train_labels = np.zeros(0)
    val_images = np.zeros(0)
    val_labels = np.zeros(0)
    test_images = np.zeros(0)
    test_labels = np.zeros(0)

    def __init__(self):
        self.filename = get('DATA.CSV_PATH')
        self.data_stored = False

    def get_images_labels(self, matrix):
        image_side = get("TRAIN.IMAGE_SIDE")
        pupil_side = get("TRAIN.PUPIL_SIDE")

        images = matrix[:, :(3 * image_side ** 2 + 6 * pupil_side ** 2)]
        labels = matrix[:, -2:]

        return images, labels

    def read_csv(self):
        df = pandas.read_csv(self.filename)
        mat = df.as_matrix()
        train_mat = mat[:get('DATA.TRAIN_NUM')]
        val_mat = mat[get('DATA.TRAIN_NUM') : get('DATA.TRAIN_NUM') + get('DATA.VAL_NUM')]
        test_mat = mat[-get('DATA.TEST_NUM'):]
        self.train_images, self.train_labels = self.get_images_labels(train_mat)
        self.val_images, self.val_labels = self.get_images_labels(val_mat)
        self.test_images, self.test_labels = self.get_images_labels(test_mat)
        self.data_stored = True

    def balance_classes(self, images, labels, count=5000):
        balanced_images, balanced_labels = [], []
        unique_labels = set(labels)
        for l in unique_labels:
            l_idx = np.where(labels == l)[0]
            l_images, l_labels = images[l_idx], labels[l_idx]
            # Consistent resampling to facilitate debugging
            resampled_images, resampled_labels = resample(l_images,
                                                          l_labels,
                                                          n_samples=count,
                                                          random_state=0)
            balanced_images.extend(resampled_images)
            balanced_labels.extend(resampled_labels)
        balanced_images = np.array(balanced_images)
        balanced_labels = np.array(balanced_labels)
        print('---Shuffled images shape: {}'.format(balanced_images.shape))
        print('---Shuffled labels shape: {}'.format(balanced_labels.shape))
        assert(len(balanced_images) == len(balanced_labels))
        shuffle_idx = np.random.permutation(len(balanced_images))
        return balanced_images[shuffle_idx], balanced_labels[shuffle_idx]

    def resize(self, images, new_size=32):
        resized = []
        for i in range(images.shape[0]):
            resized_image = imresize(images[i],
                                     size=(new_size, new_size),
                                     interp='bicubic')
            resized.append(resized_image)
        return np.array(resized)

    def preprocessed_data(self, split, dim=32, one_hot=True, balance_classes=False):
        if not self.data_stored:
            self.read_csv()
        if split == 'train':
            print_if_verbose('Loading train data...')
            images, labels = self.train_images, self.train_labels

            if balance_classes:
                images, labels = self.balance_classes(images, labels, 5000)
        elif split == 'val':
            print_if_verbose('Loading validation data...')
            images, labels = self.val_images, self.val_labels

            if balance_classes:
                images, labels = self.balance_classes(images, labels, 500)
        elif split == 'test':
            print_if_verbose('Loading test data...')
            images, labels = self.test_images, self.test_labels

            if balance_classes:
                images, labels = self.balance_classes(images, labels, 500)
        else:
            print_if_verbose('Invalid input!')
            return

        # TODO: Normalize, add dimension, one-hot encoding of labels
        # images = images.astype(np.float64)
        # for i in range(images.shape[0]):
        #     images[i] -= np.mean(images[i])
        #     images[i] /= np.linalg.norm(images[i])
        #     images[i] *= 32
        # images = np.expand_dims(images, axis = 3)

        if one_hot == True:
            new_labels = np.zeros((labels.shape[0], 4), dtype = np.float32)
            for i in range(labels.shape[0]):
                new_labels[i][int(labels[i][0])] = 1
                new_labels[i][int(labels[i][1] + 2)] = 1
            labels = new_labels

        print_if_verbose('---Images shape: {}'.format(images.shape))
        print_if_verbose('---Labels shape: {}'.format(labels.shape))
        print images
        print labels
        return images, labels


if __name__ == '__main__':
    data = PUPILS2017()
    images, labels = data.preprocessed_data('train')  # 'train' or 'val' or 'test'
