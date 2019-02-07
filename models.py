from keras2tf import *

class ModelManager:
    
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.models = {
                       '5conv':self.get_5conv_model}


    def get_5conv_model(self):
        input_shape = (224,224,3)

        input_ph = Input(input_shape)
        x = self._conv(input_ph, 32,(5,5)) # -> 112, 112, 32
        x = self._conv(x, 64, (3,3)) # -> 56, 56, 64
        x = self._conv(x, 128, (3,3)) # -> 28, 28, 128
        x = self._conv(x, 256, (3,3)) # -> 14, 14, 256
        x = self._conv(x, 512, (3,3)) # -> 7, 7, 512

        return model, input_shape, output_shape, preprocess_input



    def _conv(self, x, filters, kernel_size):
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = MaxPool2D((2,2),(2,2),padding='same')(x)
        x = BatchNormalization()(x)
        return x
