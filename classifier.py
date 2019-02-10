import os
import keras
import numpy as np
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import (BatchNormalization, Conv2D, Flatten, Input,
                          MaxPool2D, Reshape)
from keras.models import Sequential

from batch_generator import BatchGenerator
from models import ModelManager

flags = tf.app.flags
FLAGS = flags.FLAGS


class Classifier:

    def __init__(self):
        self._build()
        self.generator = BatchGenerator()

    def _build(self):
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, 10])
        self.dropout_rate = tf.placeholder_with_default(
            1., shape=None, name='dropout_rate_ph')
        self.learning_rate = tf.placeholder_with_default(
            0.0001, shape=None, name='learning_rate')
        self.avr_loss = tf.placeholder(
            dtype=tf.float32, shape=None, name='avr_loss')
        self.avr_accuracy = tf.placeholder(
            dtype=tf.float32, shape=None, name='avr_accuracy')
        self.model_manager = ModelManager(self.dropout_rate)

        self.input_placeholder, self.output = self.model_manager.models[FLAGS.model]()
        #self.output = tf.nn.softmax(output_kernel)

        self.global_step = tf.Variable(0, trainable=False)

        self.loss = tf.losses.softmax_cross_entropy(self.label_placeholder, self.output)


        _, self.accuracy = tf.metrics.accuracy(
            tf.argmax(self.label_placeholder, axis=1), tf.argmax(self.output, axis=1))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss, global_step=self.global_step)

        with tf.name_scope('tensorboard_scalars'):
            self.scalar_step = tf.summary.scalar('step', self.global_step)

            scalar_loss = tf.summary.scalar(
                'loss', self.loss)
            scalar_accuracy = tf.summary.scalar(
                'accuracy', self.accuracy)

            self.scalars_irl = tf.summary.merge([scalar_loss, scalar_accuracy])

            scalar_avr_loss = tf.summary.scalar(
                'scalar_avr_loss', self.avr_loss)           
            scalar_avr_accuracy = tf.summary.scalar(
                'scalar_avr_accuracy', self.avr_accuracy)

            self.scalar_avr = tf.summary.merge([scalar_avr_loss, scalar_avr_accuracy])

    def train(self):
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.saver = tf.train.Saver()
            train_writer, valid_writer, avr_train_writer, avr_valid_writer  = self._get_writers(sess)
            try:
                lr = FLAGS.lr
                for epoch in range(FLAGS.epochs):
                    
                    avr_accuracy = []
                    avr_loss = []
                    val_avr_accuracy = []
                    val_avr_loss = []

                    for i in range(self.generator.get_iters_count()):
                        _, accuracy, loss, summary_str = self._train_on_batch(
                            sess, [self.train_step,                                   
                                   self.accuracy,                       
                                   self.loss,
                                   self.scalars_irl], FLAGS.dropout, lr)
                        train_writer.add_summary(
                            summary_str, i + epoch * self.generator.get_iters_count())
                       
                        #print('acc:', accuracy, 'loss:', loss)
                        avr_accuracy.append(accuracy)
                        avr_loss.append(loss)
                    train_steps_count = len(avr_loss)
                    
                    

                    train_avr = sess.run([self.scalar_avr], feed_dict={self.avr_accuracy:sum(avr_accuracy)/train_steps_count,
                                                                       self.avr_loss:sum(avr_loss)/train_steps_count,
                                                                       })[0]
                    avr_train_writer.add_summary(
                        train_avr, epoch)



                    for i in range(self.generator.get_val_iters_count()):
                        accuracy, loss, summary_str = self._train_on_batch(
                            sess, [self.accuracy,                       
                                    self.loss,
                                    self.scalars_irl], 1, lr, 'valid')
                        print('validation acc:', accuracy, 'loss:', loss)
                        valid_writer.add_summary(
                            summary_str, i + epoch * self.generator.get_val_iters_count())                       
                        val_avr_accuracy.append(accuracy)
                        val_avr_loss.append(loss)
                    val_steps_count = len(val_avr_loss)
                    val_avr = sess.run([self.scalar_avr], feed_dict={self.avr_accuracy: sum(val_avr_accuracy)/val_steps_count,
                                                                     self.avr_loss: sum(val_avr_loss)/val_steps_count,
                                                                     })[0]
                    avr_valid_writer.add_summary(
                        val_avr, epoch)
                    lr *= FLAGS.decay
                    self._save(sess, epoch)
            except KeyboardInterrupt:
                self._save(sess, 10000)

    def _get_writers(self, sess):
        train_writer = tf.summary.FileWriter(
            'logs/{}_{}_{}/train'.format(self.__class__.__name__, FLAGS.batch_size, FLAGS.info), sess.graph)
        valid_writer = tf.summary.FileWriter(
            'logs/{}_{}_{}/val'.format(self.__class__.__name__, FLAGS.batch_size, FLAGS.info))
        avr_train_writer = tf.summary.FileWriter(
            'logs/{}_{}_{}/avr_train'.format(self.__class__.__name__, FLAGS.batch_size, FLAGS.info))
        avr_valid_writer = tf.summary.FileWriter(
            'logs/{}_{}_{}/avr_val'.format(self.__class__.__name__, FLAGS.batch_size, FLAGS.info))
        return train_writer, valid_writer, avr_train_writer, avr_valid_writer 

    def _train_on_batch(self, sess, eval, dropout, lr, mode='train'):
        try:
            batch_x, batch_y = self.generator.get_batch(mode)
        except:
            print(f'Problem with batch in {mode} part.')
            batch_x, batch_y = self.generator.get_batch(mode)

        result = sess.run(eval, feed_dict={self.input_placeholder: batch_x,                               
                                           self.label_placeholder: batch_y,
                                           self.dropout_rate: dropout,
                                           self.learning_rate: lr})
        return result

  

    def _save(self, sess, epoch):
        path = "sessions/{}_{}_{}/graph.ckpt".format(self.__class__.__name__,  FLAGS.batch_size, FLAGS.info)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = self.saver.save(
        sess, path, epoch)
        print("Model saved in path: %s" % save_path)
