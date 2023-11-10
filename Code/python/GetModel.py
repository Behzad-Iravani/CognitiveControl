# -*- coding: utf-8 -*-
'''
Created on Wed Nov 23 13:26:30 2022
This script is part of the main analysis for CNN

@author: neda kaboodvand and behzad iravani
Email:
n.kaboodvand@gmail.com
behzadiravani@gmail.com
'''
##-----------adding libraries-----------##
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.utils import get_custom_objects
##------------------------------------##
def linear_act(x):
    '''
    linear_act defines the saturted linear activation function
    the convolutional layers using Keras backend,
    with saturations at threholds at -1 and 1
           1    for x>=1
    f(x) = x
           -1   for x<=-1
    '''
    return K.minimum(K.maximum(x, -1),1)
get_custom_objects().update({'custom_linear': keras.layers.Activation(linear_act)}) # updating keras with the new activation function

def get_model(n_times = None, n_node = None, l1_rate = 1e-4,
              l2_rate = 1e-2, dropout_rate = 0.50,
              n_classes = None, input_shape = (None, 726, 19)):
    '''

    :param n_times:
    :param n_node:
    :param l1_rate: value for l1 normaliztion
    :param l2_rate: value for l2 normaliztion
    :param dropout_rate: value for drop out rate
    :param n_classes: integer number of classes
    :param input_shape: a tupil containing shape of the input
    :return:
    '''
    initializer = keras.initializers.GlorotNormal(seed=1)
    inputs = keras.Input((n_times, n_node),
                       name = 'Input') # batch timestep channels filters 

  
    CONVlayer1 = keras.layers.Conv1D(filters = 1, kernel_size=(32), strides=(1),
                          padding='valid',
                          activation = keras.layers.Activation(linear_act, name = "coustom_linear"),
                          name = 'conv1d_1',
                          kernel_initializer=initializer,
                          input_shape = (726,19),
                          kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate), use_bias=True)




    GlobalMax    = keras.layers.GlobalMaxPool1D()

    BatchNorm = keras.layers.BatchNormalization(scale=False, center=False, name = 'batch_normalization_1')
  
    # ------ Time domian cell
    x = CONVlayer1(inputs)
    x = tf.keras.layers.Lambda(lambda xi: tf.experimental.numpy.moveaxis(
      tf.cast(
        tf.signal.rfft(tf.experimental.numpy.moveaxis(xi,1,-1)),
             tf.float64)
       ,1,-1)
       )(x)
    x = GlobalMax(x)
    x = BatchNorm(x) #
    # flatten
    x = keras.layers.Flatten(name =  'flatten_1' )(x)
    #x = keras.layers.Dropout(dropout_rate, name = 'dropout_1')(x)
    outputs = keras.layers.Dense(units=n_classes, activation="softmax",
                               kernel_initializer=initializer,
                               name = 'dense_1',
                               kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate))(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="ConflictError")

    return model
if __name__ == '__main__':
    get_model()
