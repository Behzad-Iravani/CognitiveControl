# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 15:11:08 2022
This script performs Grad-CAM on the fitted CNN, after trainig the ntework using the Main script, run this script to perfrom
Grad-CAM and occlusion experiments.

@author: neda kaboodvand and behzad iravani
n.kaboodvand@gmail.com
behzadiravani@gmail.com

"""
# .............. Import libraries .................
import numpy as np
from scipy.io import loadmat, savemat
#------------------------------------------------
import tensorflow as tf
from tensorflow import keras

#import tensorflow_addons as tfa
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as f1
from sklearn.metrics import recall_score as  recall
from sklearn.metrics import precision_score as precesion
import collections

# plotting ---------------------
import matplotlib.pyplot as plt
# scipy ------------------------
from scipy.io import loadmat 
from scipy import stats
from scipy.io import savemat
# native ------------------------
from GetModel import get_model
from GetModel import linear_act
# --------------------------------------------------
def normalize_(signals):
    # normalize_ is a local function that normalizes the input to [-1,1]
    # input:
    #       signals: a numpy array
    # output:
    #       s:  unity normalized input
    # ------------------------------------------
    sz = signals.shape # get the input size
    s  = np.empty(sz) # initialize the output s with empty
    for items in range(len(signals)):
        s[items,:,:] = ((np.squeeze(signals[items,:,:]) -
                          np.repeat(np.expand_dims(np.mean(signals[items,:,:], axis = 1), axis = 1),sz[2], axis = 1))/np.repeat(np.expand_dims(np.max(np.squeeze(signals[items,:,:]), axis = 1)
                          -np.min(np.squeeze(signals[items,:,:]), axis = 1), axis = 1), sz[2], axis = 1))
    return s

# loading the data
path2project = "F:\Projects\Sweden\Conflict_Error\Pycharm"
data = loadmat(
    'F:\Projects\Sweden\Conflict_Error\data_incong_hit_err.mat')
signals = np.array(data['data'][0, 0]['signalPreP'])
#labels  = [unicodedata.normalize('NFKD', str(data['data'][0, 0]['label'][items][0])).encode('ascii', 'ignore') for items in range(len(data['data'][0, 0]['label']))] # 2:con 3:incon
labels  = [str(data['data'][0, 0]['label'][items][0]) for items in range(len(data['data'][0, 0]['label']))]
times   = np.array(data['data'][0, 0]['time'])
times   = times[times<=-.05]
del data # relase memory
# loading the partitioned data
data = loadmat("data_partitioned_prepratory.mat")
# ................ load the CNN ................
n_t , n_n = 726, 19
INPUT_PATCH_SIZE = (n_t, n_n)
inputs = keras.Input(shape=INPUT_PATCH_SIZE, name='Incon_Con_hit')
with keras.utils.custom_object_scope({'Activation': keras.layers.Activation(linear_act),
                                      'linear_act': linear_act}):
    #Model = get_model(n_times = n_t, n_node = n_n, n_classes = 2, input_shape = (None, n_t, n_n))
    #Model.load_weights("weights-improvement-60-0.65.hdf5")
    Model = keras.models.load_model('Model.hdf5') # load the weights
    Model.summary()
# -----------------------------------------
x_test_r = data['x_test_r']
y_test_r = data['y_test_r']
y_test_int = np.argmax(y_test_r, axis = 1)

'''Model.compile(
    loss=keras.losses.binary_crossentropy,
    metrics=["acc","Recall", "Precision", "AUC"],
)'''
# ................... create tha Grad-CAM model ...........
TestS = Model.evaluate(x_test_r, y_test_r)
grad_model = keras.models.Model([Model.inputs], [Model.get_layer('conv1d_1').output,
                                                 Model.output])
grad_model.summary() # print out the GradCam model paramteres
# --------------------------------------------------------
CAM_ = [] # initizalize the CAM_ with list that stores the Grad-CAM scores
W_   = [] # initizalize the W with list that stores the weights
for DataIndex in range(len(x_test_r)): #loop over all the test samples
  print(f'DataIndex = {DataIndex}')
  xt, CLASS_INDEX = np.copy(x_test_r[DataIndex]), np.copy(y_test_int[DataIndex])
  #xt = np.moveaxis(xt, 1, -1)
  xt = tf.expand_dims(xt, axis = -1)
  xt = tf.expand_dims(xt, axis = 0)
  
  # Compute GRADIENT
  with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(xt)
    loss = predictions[:, CLASS_INDEX]

  # Extract filters and gradients
  output = conv_output[0]
  grads = tape.gradient(loss, conv_output)[0]
  # Average gradients spatially
  weights = tf.reduce_mean(grads, axis=(0))
  # Build a ponderated map of filters according to gradients importance
  cam = np.zeros(Model.get_layer('conv1d_1').output.shape[1:2], dtype=np.float32) # changed from 0:1 to 0:2
  #cam = np.expand_dims(cam, axis = -1)
  #cam = np.repeat(cam, len(x_test_r), axis = -1)
  for index, w in enumerate(weights):
    cam += w * output[:, index]
  CAM_.append(cam)
  W_.append(weights[:].numpy())

Grad_CAM = np.array(CAM_)

'''
Grad_CAM = np.mean(Grad_CAM, axis = 0)/(np.std(Grad_CAM, axis = 0)/np.sqrt(len(CAM_)))
Szg = Grad_CAM.shape
Grad_CAM = np.interp(np.linspace(1,Szg[0],n_t), np.linspace(1,Szg[0],Szg[0]), Grad_CAM)

N = 5
plt.plot(times[0,:], np.convolve(np.transpose(Grad_CAM), np.ones(N)/N, mode = 'same'))

plt.xlabel('Time')
plt.ylabel('Grad-CAM')
plt.show()
'''
# save gradCAM results as .mat file
save_grads_cam = {"label" : y_test_int,
 "CAMS" : Grad_CAM ,
 "Weights": W_}
savemat(path2project +'/grads_data_time.mat', save_grads_cam)
# --------------------------------------
Class_model = keras.models.Model([Model.inputs], [Model.layers[-1].output])
pred = Class_model.predict(x_test_r)
epochs = 100
step_size = 1.

for CLASS_INDEX in range(2):
    # Initiate random noise
    Features = []
   
    for filter_index in range(1):
        # Initiate random noise
        #input_data = np.random.random((1, n_t, n_n))

        input_data = x_test_r[pred[:,CLASS_INDEX]
                                        == np.max(pred[:,CLASS_INDEX]),:,:]
        #input_data = normalize_(input_data)

        # Cast random noise from np.float64 to tf.float32 Variable
        input_data = tf.Variable(tf.cast(input_data, tf.float32))

        # Iterate gradient ascents
        loss_value = []
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_data)
                loss_value = tf.reduce_mean(conv_outputs[:, :,filter_index])
                grads = tape.gradient(loss_value, input_data)

        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_data.assign_add(normalized_grads * step_size)
        Features.append(input_data[0].numpy())

    savemat(path2project +f'/Features{CLASS_INDEX}.mat',{"Features": Features})


plt.plot(times, stats.zscore(input_data[0].numpy(),axis = 0))
plt.xlim([-.4,-.1])

filters, biases = Model.get_layer('conv1d_1').weights
savemat(path2project +'/Kernel_layer.mat',{"Kernel": filters[:,:,:].numpy(),
                              "Biases": biases[:].numpy()})

# normalize filter values to 0-1 so we can visualize them
model1 = keras.models.Model([Model.inputs], [Model.get_layer('conv1d_1').output])
model2 = keras.models.Model([Model.inputs], [Model.get_layer('lambda').output])
OUT1 = [];
OUT2 = [];
for DataIndex in range(len(x_test_r)): #len(x_test))
  xt, CLASS_INDEX = np.copy(x_test_r[DataIndex]), np.copy(y_test_int[DataIndex])
  xt = tf.expand_dims(xt, axis = -1)
  xt = tf.expand_dims(xt, axis = 0)
  
  conv_output = model1(xt)
  OUT1.append(conv_output)  
  
  conv_output = model2(xt)
  OUT2.append(conv_output) 

  
savemat(path2project +'/activation.mat',{"Activation1": OUT1,
                                           "Activation2": OUT2})

epochs = 10
step_size = 1e-2
MX_input = []
# finding the representative sample for each class

for CLASS_INDEX in range(2):
  # Initiate random noise
  input_data = x_test_r[pred[:,CLASS_INDEX]
                                  == np.max(pred[:,CLASS_INDEX]) ,:,:]#2*np.random.random((1,n_t, n_n))-1
  #input_data = normalize_(input_data)

  # Cast random noise from np.float64 to tf.float32 Variable
  input_data = tf.Variable(tf.cast(input_data, tf.float32))

# Iterate gradient ascents
  loss_value = []
  for _ in range(epochs):
    with tf.GradientTape() as tape:
      tape.watch(input_data)
      predictions = Class_model(input_data)
      loss = tf.reduce_mean(predictions[:, CLASS_INDEX])
      #print(loss)
    grads = tape.gradient(loss, input_data)[0]
    #print(grads)
    normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
    input_data.assign_add(tf.expand_dims(normalized_grads, axis = 0) * step_size)
  MX_input.append(input_data[0].numpy())
  
savemat(path2project +'/Maximized_class.mat',{"Class_mx": MX_input})
plt.plot(MX_input[1])
# occlusion experiment over time
win_length = 32 # the length of occlusion window
stride     = 1  # step for sliding occulsion

MAE = []     # initialize list MAE that stores the mean absolute error
MAE_VAR = [] # initialize list MAE_VAR that stores the varaince of the mean absolute error
ACC = []
F1 =[];
R =[]
P = []
for win in range(32, np.shape(times)[0], 8):#
    print(f'Occlusion window{win}')

    xt, CLASS_INDEX = np.copy(x_test_r), np.copy(y_test_int)
    xt[:,win - win_length:win, :] = np.zeros(( np.shape(xt)[0], win_length, np.shape(xt)[-1]))
    prediction = Model(xt)
    f1_ = f1(CLASS_INDEX, tf.argmax(prediction, axis=-1))

    MAE.append(np.mean(np.array(prediction), axis=0))
    MAE_VAR.append(np.var(np.array(prediction), axis=0))
    ACC.append(Model.evaluate(xt, CLASS_R)[1])
    F1.append(f1_)
    R.append(recall(CLASS_INDEX, tf.argmax(prediction, axis=-1)))
    P.append(precesion(CLASS_INDEX, tf.argmax(prediction, axis=-1)))
    

savemat(path2project +'/MeanAverageError.mat',{"MAE_AVG": np.array(MAE),
                                  "MAE_VAR": np.array(MAE_VAR),
                                  "MAE_DF": len(x_test_r),
                                  "ACC_AVG": np.array(ACC),
                                  "F1": np.array(F1),
                                  "Recall": np.array(R),
                                  "Precicion": np.array(P)
                                               })

plt.plot(1-np.array(MAE))

# occlusion experiment over channel
MAE = []
MAE_VAR = []
ACC = []
F1 =[];
R =[]
P = []
for chan in range(np.shape(x_test_r)[-1]):# loop over channel
    print(f'Occlusion chan{chan}')
    xt, CLASS_INDEX, CLASS_R = np.copy(x_test_r), np.copy(y_test_int), np.copy(
        y_test_r)
    xt[:,:, chan] = np.zeros((np.shape(xt)[0],np.shape(xt)[1]))
    prediction  = Model(xt)
    f1_ = f1(CLASS_INDEX, tf.argmax(prediction, axis = -1))

    MAE.append(np.mean(np.array(prediction), axis = 0))
    MAE_VAR.append(np.var(np.array(prediction), axis = 0))
    ACC.append(Model.evaluate(xt, CLASS_R)[1])
    F1.append(f1_)
    R.append(recall(CLASS_INDEX, tf.argmax(prediction, axis = -1)))
    P.append(precesion(CLASS_INDEX, tf.argmax(prediction, axis = -1)))
savemat(path2project +'/MeanAverageError_chan.mat',{"MAE_AVG": np.array(MAE),
                                  "MAE_VAR": np.array(MAE_VAR),
                                  "MAE_DF": len(x_test_r),
                                  "ACC_AVG": np.array(ACC),
                                  "F1": np.array(F1),
                                  "Recall":np.array(R),
                                  "Precicion":np.array(P)
                                                    })
plt.bar(range(len(MAE)),1-np.array(MAE))
print('$END')
# $ END

'''
metric = tfa.metrics.F1Score(num_classes= n_classes, threshold=0.5, average = 'weighted')

metric.update_state(keras.utils.to_categorical(y_test_int, num_classes=n_classes)
, Model(x_test_r))
result = metric.result()
result.numpy()
'''
