import os
import random
from random import sample

import cv2
import numpy as np
import tensorflow as tf

from data_augmentation import DataAugmentation

flags = tf.app.flags
FLAGS = flags.FLAGS


class BatchGenerator:

    def __init__(self, path2data=FLAGS.dataset_data):
        path2data = os.path.abspath(path2data)
        self.data = np.array(os.path.join(path2data, 'data.npy'))
        self.labels = np.array(os.path.join(path2data, 'labels.npy'))
        split = int(len(self.data)*0.7)
        test_split = int(len(self.data)*0.1)
        self.val_data = self.data[split:split+test_split]
        self.val_labels = self.labels[split:split+test_split]
        self.test_data = self.data[split+test_split:]
        self.test_labels = self.labels[split+test_split:]
        self.data = self.data[:split]
        self.labels = self.labels[:split]

        self.d = {'train': (self.data, self.labels), 
                  'val': (self.val_data, self.val_labels), 
                  'test': (self.test_data, self.test_labels)}

        self.iters = self.get_iters_count()
        self.val_iters = self.get_val_iters_count()
        self.test_iters = self.get_test_iters_count()
        self.index = 0
        self.val_index = 0
        self.test_index = 0

        self.data_augmentation = DataAugmentation()

    def get_iters_count(self):
        return len(self.data)//FLAGS.batch_size

    def get_val_iters_count(self):
        return len(self.val_data)//FLAGS.batch_size

    def get_test_iters_count(self):
        return len(self.test_data)//FLAGS.batch_size

    def _get_batch(self, index, mode='train'):      
        batch_x, batch_y = self.d[mode]
        batch_x = batch_x[index *FLAGS.batch_size: (index+1)*FLAGS.batch_size]
        batch_y = batch_y[index *FLAGS.batch_size: (index+1)*FLAGS.batch_size]
        return batch_x, batch_y

    def next_batch(self):
        while 1:
            if (self.index + 1) * FLAGS.batch_size >= len(self.data) - FLAGS.batch_size:
                self.index = 0
            try:
                ret = self._get_batch(self.index)
            except:
                print("L1 - data error - this shouldn't happen - try next batch")
                self.index += 1
                ret = self._get_batch(self.index)

            self.index += 1
            yield ret
            #return ret

    def next_val_batch(self):
        while 1:
            if (self.val_index + 1) * FLAGS.batch_size >= len(self.val_data) - FLAGS.batch_size:
                self.val_index = 0
            try:
                ret = self._get_batch(self.val_index)
            except:
                print("L1 - data error - this shouldn't happen - try next batch")
                self.val_index += 1
                ret = self._get_batch(self.val_index)

            self.val_index += 1
            yield ret
            #return ret

    def next_test_batch(self):
        while 1:
            if (self.test_index + 1) * FLAGS.batch_size >= len(self.test_data) - FLAGS.batch_size:
                self.test_index = 0
            try:
                ret = self._get_batch(self.test_index)
            except:
                print("L1 - data error - this shouldn't happen - try next batch")
                self.test_index += 1
                ret = self._get_batch(self.test_index)

            self.test_index += 1
            yield ret
            #return ret

    def get_batch(self, mode):
        if mode == 'train':
            return self.next_batch()
        elif mode == 'val':
            return self.next_val_batch()
        else:
            return self.next_test_batch()

