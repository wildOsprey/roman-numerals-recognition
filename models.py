from keras2tf import *
import tensorflow as tf

class ModelManager:
    
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.models = {
                       '5conv':self.get_5conv_model}


    def get_5conv_model(self):
        input_shape = (28,28,1)

        input_ph = Input(input_shape)
        x = self._conv(input_ph, 16,(3,3))
        x = self._conv(x, 32, (3,3)) 
        x = Flatten()(x)
        x = Dense(32, activation=tf.nn.relu)(x)
        output = Dense(10)(x)
        return input_ph, output



    def _conv(self, x, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(x)
        x = MaxPool2D((2,2),(2,2),padding='same')(x)
        #x = BatchNormalization()(x)
        return x
