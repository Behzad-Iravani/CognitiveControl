# -*- coding: utf-8 -*-
'''
DM is class for perfoming dynamical system modeling using tvb for cognitive-error control project
Authors: Neda Kaboodvand and Behzad Iravani
   n.kaboodvand@gmail.com
   behzadiravani@gmail.com
Palo Alto, January 2023
'''
import numpy as np

from scipy import fft

import matplotlib.pyplot as plt

from tvb.simulator.lab import *
from tvb.datatypes.time_series import TimeSeriesRegion
import tvb.simulator.plot.power_spectra_interactive as ps_int

class modelDM:
    def __init__(self, data = None, mu = None ,
                 v0 = None, jrm = None):
        self.data = data
        self.mu = mu
        self.v0 = v0
        self.jrm = jrm


    @classmethod
    def get_dc_model(cls, mu, v0):
        jrm = models.JansenRit(mu=mu, v0=v0)
        return cls(mu= mu, v0 = v0, jrm = jrm)


    def config_sim(self, coupling, conn, sigma):
        # create the model
        mdl = self.get_dc_model(self.mu, self.v0)

        sim = simulator.Simulator(
        model=mdl.jrm,
        connectivity=conn,
        coupling=coupling,
        integrator=integrators.HeunStochastic(dt= 1 , noise=noise.Additive(nsig=sigma)),
        monitors= (monitors.Raw(),monitors.ProgressLogger(period=1e2),),
        simulation_length=2e3, # in ms
        ).configure()
        return sim

    def get_timeserie(self, sim, tavg, time, conn):
        # Discarded first seconds to avoid initial transient dynamics.
        tavg = tavg[time>500]  #in ms

        # Build a TimeSeries Dataype.
        tsr = TimeSeriesRegion(connectivity=conn,
                               data=tavg,  # in TVB 4D format
                               sample_period=sim.monitors[0].period,
                               time = time[time>500]
                               )  # in ms
        tsr.configure()
        return tsr
'''
    def plot_power(self, tsr, sim, state, region):
        # Configure and ...
        dat = tsr.data.reshape(tsr.shape)
        dat = np.squeeze(dat[:,state,region,0])
        Sxx =  fft.fft(dat)
        n = Sxx.size
        timestep = tsr.sample_rate**-1
        freq = np.fft.fftfreq(n, d=timestep)
        plt.plot(freq,np.squeeze(np.nanmean(Sxx, axis=-1)))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.xlim([0, 250])
        plt.show()

'''
# main dunder
if __name__ == '__main__':
    modelDM










