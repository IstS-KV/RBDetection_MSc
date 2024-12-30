import sklearn
import tensorflow as tf
import numpy
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import L1, L2, L1L2


class cnn_phase1(object):
    def __init__(self, dropout, is_train):

        self.is_train = is_train
        self.dropout = dropout
        
    @staticmethod
    def masking_layer(x0, mask_val = -1):
        return layers.Masking(mask_value=mask_val)(x0)

    @staticmethod 
    def cnn2d_layer(x0, n_filters, kernel_size, strides, dilation_rate, padding, regularizer):
        if regularizer == 'l1':
            return layers.Conv2D(n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, activation='relu', padding=padding, kernel_regularizer=L1(0.01))(x0)
        elif regularizer == 'l2':
            return layers.Conv2D(n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, activation='relu', padding=padding, kernel_regularizer=L2(0.01))(x0)
        elif regularizer == 'l1l2':
            return layers.Conv2D(n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, activation='relu', padding=padding, kernel_regularizer=L1L2(l1=0.01, l2=0.01))(x0)
        else:
            return layers.Conv2D(n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, kernel_regularizer=L2(0.01))(x0)

    @staticmethod 
    def batch_norm_block(x, is_train, decay=0.999, epsilon=1e-5):
        return layers.BatchNormalization(momentum=decay, epsilon=epsilon, trainable=is_train)(x)

    @staticmethod 
    def cnn2d_with_batch(x, n_filters, kernel_size, strides, dilation_rate, padding, is_train, regularizer=False):
        output1 = cnn_phase1.cnn2d_layer(x, n_filters, kernel_size, strides, dilation_rate, padding, regularizer)
        output2 = cnn_phase1.batch_norm_block(output1, is_train)
        return output2
    
    @staticmethod
    def ResBlock(x, n_filters, kernel_size, strides, dilation_rate, padding, is_train, regularizer1='none', regularizer2='none'):
        output1 = cnn_phase1.cnn2d_with_batch(x, n_filters, kernel_size, strides=1, dilation_rate=dilation_rate, padding=padding, is_train=is_train, regularizer=regularizer1)
        output1 = cnn_phase1.cnn2d_with_batch(output1, n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, is_train=is_train, regularizer=regularizer2)

        if n_filters != x.shape[-1] or strides > 1:
            output2 = cnn_phase1.cnn2d_with_batch(x, n_filters, kernel_size=1, strides=strides, dilation_rate=1, padding='same', is_train=is_train)
        else:
            output2 = layers.Lambda(lambda x: x)(x)

        return output1 + output2 
    
    @staticmethod
    def inverted_ResBlock(x, n_filters, kernel_size, strides, dilation_rate, padding, is_train, expansion_factor, regularizer1='none', regularizer2='none'):
        # expansion
        output1 = cnn_phase1.cnn2d_with_batch(x, n_filters*expansion_factor, kernel_size, strides=1, dilation_rate=dilation_rate, padding=padding, is_train=is_train, regularizer=regularizer1)
        # depthwise convolution
        output1 = layers.DepthwiseConv2D(kernel_size = 3, strides=1, dilation_rate=dilation_rate, padding=padding, kernel_regularizer=L2(0.01))(output1)
        output1 = cnn_phase1.batch_norm_block(output1, is_train)
        # back to normal size
        output1 = layers.Conv2D(n_filters, kernel_size, strides=strides, dilation_rate=dilation_rate, padding=padding, kernel_regularizer=regularizer2)(output1)
        output1= cnn_phase1.batch_norm_block(output1, is_train)

        # skip connection
        if n_filters != x.shape[-1] or strides > 1:
            output2 = cnn_phase1.cnn2d_with_batch(x, n_filters, kernel_size=1, strides=strides, dilation_rate=1, padding='same', is_train=is_train)
        else:
            output2 = layers.Lambda(lambda x: x)(x)

        return output1 + output2
    
    @staticmethod
    def MaxPool(x, pool_size, strides):
        return layers.MaxPool(pool_size=pool_size, strides=strides)(x)


    def build_model(self, input_, residual_block_num):
        
        self.x = input_
        self.residual_block_num = residual_block_num

        o1 = cnn_phase1.masking_layer(self.x)
        o1 = cnn_phase1.cnn2d_with_batch(o1, n_filters=32, kernel_size=12, strides=1, dilation_rate = 1, padding ='same', is_train=self.is_train, regularizer='l1l2')
        o1 = cnn_phase1.cnn2d_with_batch(o1, n_filters=32, kernel_size=3, strides=2, dilation_rate = 1, padding='same', is_train=self.is_train, regularizer='l2')


        if self.dropout:
           if self.is_train:
                o1 = layers.Dropout(rate=0.5)(o1)
           else:
                o1 = layers.Dropout(rate=0)(o1)

        o1 = cnn_phase1.inverted_ResBlock(o1, n_filters=32, kernel_size=1, strides=2, dilation_rate=1, padding='same', is_train=self.is_train, expansion_factor=2, regularizer1='l2', regularizer2=L1L2(l1=0.01, l2=0.01))
        o1 = cnn_phase1.inverted_ResBlock(o1, n_filters=32, kernel_size=1, strides=1, dilation_rate=1, padding='same', is_train=self.is_train, expansion_factor=2, regularizer1='l2', regularizer2=L2(0.01))
        
        if self.dropout:
            if self.is_train:
                o1 = layers.Dropout(rate=0.5)(o1)
            else:
                o1 = layers.Dropout(rate=0)(o1)

        o1 = cnn_phase1.inverted_ResBlock(o1, n_filters=64, kernel_size=1, strides=2, dilation_rate=1, padding='same', is_train=self.is_train, expansion_factor=3, regularizer1='l2', regularizer2=L1L2(l1=0.01, l2=0.01))
        o1 = cnn_phase1.inverted_ResBlock(o1, n_filters=64, kernel_size=1, strides=1, dilation_rate=1, padding='same', is_train=self.is_train, expansion_factor=3, regularizer1='l2', regularizer2=L2(0.01))

        if self.residual_block_num == 1:
            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 1, padding = 'same', is_train=self.is_train)
            print()
        elif self.residual_block_num == 2:
            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 1, padding = 'same', is_train=self.is_train)
            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 1, padding = 'same', is_train=self.is_train)
            print()
        elif self.residual_block_num == 3:
            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 2, padding = 'same', is_train=self.is_train)

            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 4, padding = 'same', is_train=self.is_train)
            o1 = cnn_phase1.ResBlock(o1, n_filters = 64, kernel_size = 12, strides = 1, dilation_rate = 6, padding = 'same', is_train=self.is_train)
            print()
        else:
            print('No residual blocks')

        o1 = layers.Flatten()(o1)

        return o1
    
class CombinedCNNs(object):

    def __init__(self, dropout, is_train, num_res_blocks, batch_size, height, width, n_class, num_channels):
        super(CombinedCNNs, self).__init__()
        self.dropout = dropout
        self.is_train = is_train
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.n_class = n_class
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

    def build_model(self, inputs_=None):

        # define the shape of the input based on the depth (batch_size, height, width, channels)
        if inputs_ == None:
            if self.num_channels == 1:
                input_shape = [self.height, self.width, 1]
            elif self.num_channels == 2:
                input_shape = [self.height, self.width, 2]
            elif self.num_channels == 3:
                input_shape = [self.height, self.width, 3]
            else :
                print('ERROR: The num_channels parameter is out of possible range')
            inputs_ = tf.keras.Input(shape = input_shape)
        self.input = inputs_

        cnn1 = cnn_phase1(dropout=self.dropout, is_train=self.is_train)
        network= cnn1.build_model(self.input, self.num_res_blocks)

        # add logits to the output 
        logits_output = layers.Dense(self.n_class)(network)
        model = tf.keras.Model(inputs=self.input, outputs=logits_output)

        # add probabilities 
        # prob_output = layers.Dense(self.n_class)(network)
        # model = tf.keras.Model(inputs=self.inputs_, outputs=prob_output)

        return model