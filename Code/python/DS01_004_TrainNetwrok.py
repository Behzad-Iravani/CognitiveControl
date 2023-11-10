# -*- coding: utf-8 -*-
"""
This script trians the CNN on the reconstructed EEG sources that were identied using fMRI
The EEG sources were reconstructed using the eLORETA and within MATLAB scripts
The reconstructed time courses were stored in *.mat file

Created on Wed Nov 23 13:01:55 2022
@author: nedkab and behira

"""
# Import libraries
from scipy.io import loadmat 
from scipy.io import savemat 
from scipy.stats import zscore 
# ---------------------
import numpy as np
import pandas as pd
# ---------------------
from matplotlib import pyplot as plt
from matplotlib import cm
# ---------------------
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
# ----------------------
from tensorflow import keras
import tensorflow as tf
# ----------------------
import collections
# ----------------------
import unicodedata
# ----------------------
import os 
# ----------------------
from GetModel import get_model, linear_act
from smooth_ import smooth

## define functions
# normalized over -1 and 1 over time
def normalize_(signals):
    sz = signals.shape
    s = np.empty(sz)
    for items in range(len(signals)):
        s[items,:,:] = ((np.squeeze(signals[items,:,:]) -
                          np.repeat(np.expand_dims(np.mean(signals[items,:,:], axis = 1), axis = 1),sz[2], axis = 1))/np.repeat(np.expand_dims(np.max(np.squeeze(signals[items,:,:]), axis = 1)
                          -np.min(np.squeeze(signals[items,:,:]), axis = 1), axis = 1), sz[2], axis = 1))
    return s
# for reproducibility 
tf.random.set_seed(2)
np.random.seed(3)
#------------------------
# ----- data ------
path2project = "F:\Projects\Sweden\Conflict_Error\Pycharm" # path to data
# loading data
data = loadmat(
    'F:\Projects\Sweden\Conflict_Error\data_incong_hit_err.mat')
# -----------------
signals = np.array(data['data'][0, 0]['signalPreP']) # source prepratory electrophysiological activity
signals_aft = np.array(data['data'][0, 0]['signalAft']) # source post-stimulus onset electrophysological activity
#labels  = [unicodedata.normalize('NFKD', str(data['data'][0, 0]['label'][items][0])).encode('ascii', 'ignore') for items in range(len(data['data'][0, 0]['label']))] # 2:con 3:incon 
labels  = [str(data['data'][0, 0]['label'][items][0]) for items in range(len(data['data'][0, 0]['label']))]
times   = np.array(data['data'][0, 0]['time'])
rt      = np.array(data['data'][0, 0]['RT'])
sub  = [str(data['data'][0, 0]['sub'][items][0]) for items in range(len(data['data'][0, 0]['sub']))]
sub = np.transpose(sub)
# make sure that network is trained on the perpratoy activity
#signals = signals[:,:,times[0,:]<=-.05] # select times< -.05
print(signals.shape)
times = times[:,times[0,:]<=-.05]
print(times.shape)
# preprocessing the data
y  = preprocessing.LabelEncoder()
y.fit(labels)
y = y.transform(labels)
# print the number
print(collections.Counter(np.squeeze(labels)))
print(collections.Counter(y))
s = normalize_(signals)
#s = np.expand_dims(s,axis = 1)
sz = s.shape
print(s.shape)
plt.plot(np.transpose(times),np.transpose(np.squeeze(s[5,:,:])))

# Train/Test/Validation Set Splitting
x_train, x_rem, y_train_int, y_rem_int, rt_train, rt_rem, sub_train, sub_rem   \
    = train_test_split(s, y, rt, sub, test_size=0.4, random_state=2, stratify=y) # increased test/validation from .4
x_valid, x_test, y_valid_int, y_test_int, rt_valid, rt_test, sub_valid, sub_test  \
    = train_test_split(x_rem, y_rem_int, rt_rem, sub_rem, test_size=0.25, random_state=2, stratify=y_rem_int)
# free the memory by removing some varaibles
del x_rem, y_rem_int, rt_rem, sub_rem
# ----------------------------------------
# Training data
sz = x_train.shape
x_train_r = np.moveaxis(x_train, 1,-1)
# Validation data
sz = x_valid.shape
x_valid_r = np.moveaxis(x_valid, 1,-1)
# ----------------------------------------
sz = x_test.shape
x_test_r =  np.moveaxis(x_test, 1,-1)
print(np.shape(x_train_r))
print(np.shape(x_valid_r))
print(collections.Counter(y_train_int))
print(collections.Counter(y_valid_int))

# Adding class weights
class_weights = class_weight.compute_class_weight(class_weight = "balanced",
                                                  classes = np.unique(y_train_int),
                                                  y = y_train_int)
class_weights = dict(zip(np.unique(y_train_int), class_weights)),
print(class_weights)

# plot a random sample 
plt.plot(np.transpose(times),np.squeeze(x_train_r[589,:,:]))

n_classes = 2
y_train_r = keras.utils.to_categorical(y_train_int, num_classes=n_classes)
y_valid_r = keras.utils.to_categorical(y_valid_int, num_classes=n_classes)
y_test_r  = keras.utils.to_categorical(y_test_int, num_classes=n_classes)
# saving the partitioned data
savemat('data_partitioned_prepratory.mat', {'x_train_r': x_train_r,
                                      'x_valid_r': x_valid_r,
                                      'x_test_r': x_test_r,
                                      'y_train_r': y_train_r,
                                      'y_valid_r': y_valid_r,
                                      'y_test_r': y_test_r,
                                      'rt_train': rt_train,
                                      'rt_valid': rt_valid,
                                      'rt_test': rt_test,
                                      'sub_train': sub_train,
                                      'sub_valid': sub_valid,
                                      'sub_test': sub_test
                                 })
# ------------------FITTING THE MODLE---------------------
n_t , n_n = 726, 19
INPUT_PATCH_SIZE = (n_t, n_n)
inputs = keras.Input(shape=INPUT_PATCH_SIZE, name='Incon_Con_hit')
Model = get_model(n_times = n_t, n_node = n_n, n_classes = 2, input_shape = (None, n_t, n_n), dropout_rate = .1)
Model.summary()

@tf.function

def train_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis= -1)
    return volume, label
def valid_preprocessing(volume, label):
    volume = tf.expand_dims(volume, axis= -1)
    return volume, label

batch_size   = 25 # 
train_loader = tf.data.Dataset.from_tensor_slices((x_train_r, y_train_r))
valid_loader = tf.data.Dataset.from_tensor_slices((x_valid_r, y_valid_r))


train_dataset = (
    train_loader.shuffle(len(x_train_r))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

valid_dataset = (
    valid_loader.shuffle(len(x_valid_r))
    .map(valid_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

# Compile model.
learning_rate = 2e-4
epochs = 100

# Optimizer
initial_learning_rate = learning_rate
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.90, staircase=True
)

Model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                    mode='min')

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)  # 5
# Train the model, doing validation at the end of each epoch

history = Model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    shuffle=True,
    class_weight=class_weights[0],
    verbose=1,
    callbacks=[checkpoint_cb, early_stopping_cb],
)
# saving model
keras.models.save_model(Model, 'Model.hdf5')
Metric = smooth(history)
# savinf the metrics
savemat(path2project + '/metric.mat', {'Metric': Metric})

plt.close
fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()
col = cm.get_cmap('tab20b')
for i, metric in enumerate(["acc", "loss"]):
  ax[i].plot(Metric[metric][:,1], color = col(.01), linewidth = 2.0)
  ax[i].plot(Metric["val_" + metric][:,1], color = col(.45), linewidth = 2.0)
  ax[i].set_title("Model {}".format(metric))
  ax[i].set_xlabel("Epochs")
  ax[i].set_ylabel(metric)
  ax[i].legend(["Training", "Validation"], frameon=False, fontsize = 14)
  right_side = ax[i].spines["right"]
  right_side.set_visible(False)
  top_side = ax[i].spines["top"]
  top_side.set_visible(False)
  for axis in ['bottom','left']:
    ax[i].spines[axis].set_linewidth(2.0)
  if i==0:
    ax[i].set_yticks(np.linspace(.45,.75,3))
    ax[i].set_ylim([.45, .75])
  else:
    ax[i].set_yticks(np.linspace(.60, 1.15,3))
    ax[i].set_ylim([.60, 1.15])
  for item in ([ax[i].xaxis.label, ax[i].yaxis.label]):
    item.set_fontsize(16)
  for item in (ax[i].get_xticklabels() + ax[i].get_yticklabels()):
    item.set_fontsize(14)
  ax[i].title.set_fontsize(20)
#plt.show()  
plt.savefig(path2project + "/ACC_edited.svg")


# Generate generalization metrics
# Validation
ValidationScores = Model.evaluate(x_valid_r, y_valid_r, verbose=0)
# Test
TestScores = Model.evaluate(x_test_r, y_test_r, verbose=0)
print(f'Validation score: {Model.metrics_names[0]} of {ValidationScores[0]}; {Model.metrics_names[1]} of {ValidationScores[1]*100}%')
mloss = np.min(Metric['loss'][:,1])
macc  = np.max(Metric['acc'][:,1])
with open(path2project + "/accuracy.txt", 'w') as f:
    f.write(
        f'Training score: loss of {mloss}; acc of {macc*100}% \n')
    f.write(
           f'Validation score: {Model.metrics_names[0]} of {ValidationScores[0]}; {Model.metrics_names[1]} of {ValidationScores[1]*100}% \n')  
    f.write(
        f'Test score: {Model.metrics_names[0]} of {TestScores[0]}; {Model.metrics_names[1]} of {TestScores[1]*100}% \n')
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)
hist_csv_file = path2project + '/history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#-----------
# $END
    
  