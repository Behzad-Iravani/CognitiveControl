# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:39:56 2022

@author: behira
"""

import statsmodels.api as sm
import numpy as np

def smooth(history):
    Metric = {'acc': history.history["acc"],
          'loss': history.history["loss"],
          'val_acc': history.history["val_acc"],
          'val_loss': history.history["val_loss"]
          }
    for metric in ["acc", "loss", "val_acc", "val_loss"]:
        Metric[metric] = np.array(sm.nonparametric.lowess(Metric[metric], 
                                             np.linspace(0,len(Metric[metric]),len(Metric[metric])), frac=0.1))
        
        
    return Metric