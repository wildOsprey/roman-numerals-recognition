import tensorflow as tf

def Input(input_shape):
    return tf.placeholder(tf.float32, shape=[None, *input_shape])

def Reshape(shape, name=None):
    def _Reshape(x):
        return tf.reshape(x, shape, name=name)
    return _Reshape

def Flatten(name=None):
    def _Flatten(x):
        return tf.layers.flatten(x, name=name)

def Dense(units, **args):
    def _Dense(x):
        tf.layers.dense(x, units=units, **args)
    return _Dense

def Conv2D(filters, kernel_size, **args):
    def _Conv2D(x):
        return tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size, **args)
    return _Conv2D

def MaxPool2D(pool_size, strides_size, **args):
    def _MaxPool2D(x):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=strides_size, **args)
    return _MaxPool2D

def BatchNormalization(**args):
    def _BatchNormalization(x):
        return tf.layers.batch_normalization(x, **args)
    return _BatchNormalization

def Dropout(dropout_rate, **args):
    #Keras version of dropout rate. Rate to drop.
    def _Dropout(x):
        return tf.layers.dropout(x, 1.-dropout_rate, **args)
    return _Dropout

