# -*- coding: 'UTF-8' -*-
'''
SDCNN_Pred_OF_DM simulates the EEG-like response of Jansen rit model as a function of different
values for paramter C.

Authors: Neda Kaboodvand & Behzad Iravani

'''
from tensorflow import keras
import os
from scipy.io import savemat, loadmat


def load_mode(n_t, n_n):
    # load_mode is a local function that loads the trained SCCN model.

    from GetModel import  linear_act
    INPUT_PATCH_SIZE = (n_t, n_n)
    inputs = keras.Input(shape=INPUT_PATCH_SIZE, name='Incon_Con_hit')
    with keras.utils.custom_object_scope({'Activation': keras.layers.Activation(linear_act),
                                          'linear_act': linear_act}):
        # Model = get_model(n_times = n_t, n_node = n_n, n_classes = 2, input_shape = (None, n_t, n_n))
        # Model.load_weights("weights-improvement-60-0.65.hdf5")
        Model = keras.models.load_model('Model.hdf5')
        Model.summary()
    return Model
# load SCNN
n_t, n_n = 726, 19 # number of time points, number of regions
# load mode
SCNN = load_mode(n_t, n_n)
# path to save simulated data
result = os.listdir('simulation/BigC')
cnt = 0;
for items in result:

    if items.endswith(".mat") and not items.__contains__('args') and not items.__contains__('search')  and not items.__contains__('orederMAE'):
        num = items.split('_')[-1]
        num = num.split('.')[0]
        print(f'\t \t>> prediction {int(num)}::::::{items}')
        LS = loadmat(f'simulation/BigC/{items}')
        output = SCNN.predict(LS['dat_downsample'])
        num = items.split('_')[-1]
        num = num.split('.')[0]
        savemat(f'simulation/BigC/search_{int(num)}.mat', {'output': output})
