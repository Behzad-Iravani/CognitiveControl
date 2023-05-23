# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:26:30 2022

@author: behira
"""
from tensorflow import keras
import tensorflow as tf
from keras import backend as K
from keras.utils import get_custom_objects


def linear_act(x):
    '''y = []
    for items in x:
        
        if items<-1:
            y.append(tf.cast(-1, dtype = tf.float64))
        elif items>1:
            y.append(tf.cast(1, dtype = tf.float64))
        else:
           y.append(items)
    return tf.convert_to_tensor(y)
    y = tf.identity(x)
    for rows in range(x.shape[1]):
        y[:,rows,:] =  K.minimum(K.maximum(x[:,rows,:], -1),1)'''
    return K.minimum(K.maximum(x, -1),1)




get_custom_objects().update({'custom_linear': keras.layers.Activation(linear_act)})


def get_model(n_times = None, n_node = None, l1_rate = 1e-4,
              l2_rate = 1e-2, dropout_rate = 0.50,
              n_classes = None, input_shape = (None, 726, 19)):
    
    
  initializer = keras.initializers.GlorotNormal()
  inputs = keras.Input((n_times, n_node),
                       name = 'Input') # batch timestep channels filters 

  #x = tf.keras.layers.Lambda(lambda xi:  tf.experimental.numpy.moveaxis(
       # tf.cast(
      #  tf.signal.stft(tf.experimental.numpy.moveaxis(xi,1,-1), frame_length = 64, frame_step = 32),
     #       tf.float64)
    #,1,-1)
   # )(inputs) # fft over the inner most dimension
    #
  
  CONVlayer1 = keras.layers.Conv1D(filters = 2, kernel_size=(128), strides=(1),
                          padding='valid',
                          activation = keras.layers.Activation(linear_act, name = "coustom_linear"),
                          name = 'conv1d_1',
                          kernel_initializer=initializer,
                          input_shape = (726,19),
                          kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate), use_bias=True)
 
 
 
  CONVlayer2 = keras.layers.Conv1D(filters = 3, kernel_size=(32), strides=(1),
                          padding ='valid',
                          activation = keras.layers.Activation(linear_act, name = "coustom_linear"),
                          name = 'conv1d_2',
                          kernel_initializer=initializer,
                          input_shape = (363,4),
                          kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate), use_bias=True)
  
 
  CONVlayer3 = keras.layers.Conv1D(filters = 4, kernel_size=(8), strides=(1),
                          padding='valid',
                          name = 'conv1d_3',
                          activation= keras.layers.Activation(linear_act, name = "coustom_linear"),
                          kernel_initializer=initializer,
                          input_shape = (182,5),
                          kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate), use_bias=True)
  
  
  AVGPOOL1    = keras.layers.AveragePooling1D(pool_size=(4), strides=(4), padding='valid', name = 'avg_pool1d_1')
  AVGPOOL2    = keras.layers.AveragePooling1D(pool_size=(4), strides=(4), padding='valid', name = 'avg_pool1d_2')
  AVGPOOL3    = keras.layers.AveragePooling1D(pool_size=(4), strides=(4), padding='valid', name = 'avg_pool1d_3')
  
  #MaxPOOL    = keras.layers.MaxPooling1D(pool_size=(5), strides=(5), padding='same')  # 10 nodes X 100 time step  features
  BatchNorm = keras.layers.BatchNormalization(scale=False, center=False, name = 'batch_normalization_1')
  
    # ------ Time domian cell
  x = CONVlayer1(inputs)
  x = AVGPOOL1(x)
  x = CONVlayer2(x) 
  x = AVGPOOL2(x)
  x = CONVlayer3(x) # ------
  x = AVGPOOL3(x)

  x = BatchNorm(x) #
  # flatten
  x = keras.layers.Flatten(name =  'flatten_1' )(x)  
  x = keras.layers.Dropout(dropout_rate, name = 'dropout_1')(x)
  outputs = keras.layers.Dense(units=n_classes, activation="softmax",
                               kernel_initializer=initializer,
                               name = 'dense_1',
                               kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1_rate, l2=l2_rate))(x)

    # Define the model.
  model = keras.Model(inputs, outputs, name="ConflictError")
  return model

