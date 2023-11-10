# -*- coding: utf-8 -*-


import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


from tvb.simulator.lab import *
from scipy.signal import (decimate, resample)
from scipy.io import savemat, loadmat
from DynamicModeling import DM as DM

from os.path import exists

import matplotlib.pyplot as plt

##---------------------------##
def normalize_(signals):
    sz = signals.shape
    s = np.empty(sz)
    for items in range(len(signals)):
        s[items,:,:] = ((np.squeeze(signals[items,:,:]) -
                          np.repeat(np.expand_dims(np.mean(signals[items,:,:], axis = 1), axis = 1),sz[2], axis = 1))/np.repeat(np.expand_dims(np.max(np.squeeze(signals[items,:,:]), axis = 1)
                          -np.min(np.squeeze(signals[items,:,:]), axis = 1), axis = 1), sz[2], axis = 1))
    return s



def paralle_(DM, conn, sigma, index, val):
    print(f'\t>>Iteration {index}')
    n_t, n_n = 726, 19
    #DM.mu = np.zeros((1,))
    DM.v0 = np.zeros((19,1))



    #print(f'iteration:{it} of {4 * 2 ** 19}')
    DM.v0[0] = np.array(val[1])
    DM.v0[1] = np.array(val[2])
    DM.v0[2] = np.array(val[3])
    DM.v0[3] = np.array(val[4])
    DM.v0[4] = np.array(val[5])
    DM.v0[5] = np.array(val[6])
    DM.v0[6] = np.array(val[7])
    DM.v0[7] = np.array(val[8])
    DM.v0[8] = np.array(val[9])
    DM.v0[9] = np.array(val[10])
    DM.v0[10]= np.array(val[11])
    DM.v0[11]= np.array(val[12])
    DM.v0[12]= np.array(val[13])
    DM.v0[13]= np.array(val[14])
    DM.v0[14]= np.array(val[15])
    DM.v0[15]= np.array(val[16])
    DM.v0[16]= np.array(val[17])
    DM.v0[17]= np.array(val[18])
    DM.v0[18]= np.array(val[19])
    # -------------- get JSR model ----------#
    # DM = DM.get_dc_model(mu= DM.mu, v0= DM.v0) in config method
    # ----------------------------------------#
    sim = DM.config_sim(coupling=coupling.Linear(a=np.array([val[0]])), conn=conn,
        sigma=sigma)  # SigmoidalJansenRitval[0]
    sim.configure()
    # run sim
    (time, data),(_, _) = sim.run()
    '''plt.plot(time, data[:,0,0])
    plt.xlim([100,2e3])
    plt.ylim([-3, 30])
    plt.show()'''
    ##--- Get the time series of regions ---##
    tsr = DM.get_timeserie(sim=sim, tavg=data, time=time, conn=conn)
    dat = np.squeeze(tsr.data[:, 0, :, 0])  # frist state variable
    dat_downsample = resample(dat, n_t)
    # normalize
    dat_downsample = normalize_(np.expand_dims(dat_downsample, axis=0))

    savemat(f'simulation\BigC\simulated_{index}.mat', {'dat_downsample': dat_downsample})

    #return dat_downsample

if __name__ == '__main__':
    args = []
    pnt = 2
    if not exists('simulation\BigC\_args.mat'):
        # domain=Range(lo=3.12, hi=6.0, step=0.02),
        for val0 in np.linspace(0.0001, 0.05, 200):
            tmp = [6.0]*19;
            tmp.insert(0, val0)
            args.append(tmp)

        savemat(f'simulation\BigC\_args_BigC.mat', {'args': args})
    else:
        args = loadmat('simulation\BigC\_argsBigC.mat')
        args = args['args']



    # load Dynamic model

    DM = DM.modelDM
    DM = DM(mu=np.array(.0), v0=np.ones((19,))*3.12)

    # load conn
    conn = pickle.load(open('conn.pkl', 'rb'))
    # load sigma
    sigma = pickle.load(open('sigma.pkl', 'rb'))
    # Step 1: Init multiprocessing.Pool()
    num_cores = multiprocessing.cpu_count()
    #result = (paralle_(DM, conn, sigma, index, items) for index, items in enumerate(args))
    result = Parallel(n_jobs=num_cores)(delayed(paralle_)(DM, conn, sigma, index, items) for index, items in enumerate(args))
    #results = paralle_(DM, conn, sigma, items)
    # predicting simulation using sCNN


