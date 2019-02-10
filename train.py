import tensorflow as tf

from classifier import Classifier

flags = tf.app.flags
FLAGS = flags.FLAGS

EPOCHS = 100
VAL_RATE = 3
ADAM_LR = 0.0001
DECAY = 1.
DROPOUT_RATE = 0.7


flags.DEFINE_string('dataset_data', "dataset/",
                    'Dataset data')
flags.DEFINE_string('info', "INT20H",
                    'Info')

flags.DEFINE_integer(
    'batch_size', 32, 'Batch size to use. Default value is 256')

flags.DEFINE_string('model', '5conv', 'Default value is 5conv')
flags.DEFINE_string('augmentation', 'small', 'Default value is small')

flags.DEFINE_integer('epochs', 10, 'Epochs count.')
flags.DEFINE_float('lr', 0.001, 'Learning rate.')
flags.DEFINE_float('decay', .9, 'Decay rate.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate.')


def main(_):
    model = Classifier()
    model.train()


if __name__ == '__main__':
    tf.app.run(main=main)
