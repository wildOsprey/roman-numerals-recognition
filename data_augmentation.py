import numpy as np
import cv2
import os
import random
from random import sample
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class DataAugmentation:

    def __init__(self):
        self.full_augmentation = [self.no_action,
                                  self.vertical_flip,
                                  self.horizontal_flip,
                                  self.horizontal_vertical_flip,
                                  self.rot_90,
                                  self.rot_180,
                                  self.rot_270,
                                  ]

        self.short_augmentation = [self.no_action,
                                   self.horizontal_flip,
                                   self.rot_90,
                                   self.rot_270,
                                   ]

        self.augment = self.small_augmentation if FLAGS.augmentation == 'small' else self.strong_augmentation

    def small_augmentation(self, image):
        return self.short_augmentation[random.randint(0, len(self.short_augmentation)-1)](image)

    def strong_augmentation(self, image):
        return self.full_augmentation[random.randint(0, len(self.full_augmentation)-1)](image)

    def no_action(self, image):
        return image

    def vertical_flip(self, img):
        return cv2.flip(img, 0)

    def horizontal_flip(self, img):
        return cv2.flip(img, 1)

    def horizontal_vertical_flip(self, img):
        return cv2.flip(img, -1)

    def rot_90(self, img):
        return self.rot(img, 90)

    def rot_180(self, img):
        return self.rot(img, 180)

    def rot_270(self, img):
        return self.rot(img, 270)

    def rot(self, img, angle):
        w, h = img[0], img[1]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.)
        return cv2.warpAffine(img, M, (w, h))
