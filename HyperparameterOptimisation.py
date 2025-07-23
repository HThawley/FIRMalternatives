# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 12:30:53 2025

@author: u6942852
"""

import numpy as np 
from numba import njit
from psutil import cpu_count
from tqdm import tqdm

from Input import * 
from Fileprinter import Fileprinter
from Optimisation import ObjectiveWrapper
from mlp import MLPmodel

#%%
INIT = False
RANDOMSEED = None
BATCHSIZE = cpu_count(True) * 10 
ITERATIONS = None
EVALUATIONS = 1_000_000
assert ITERATIONS is None or EVALUATIONS is None
if ITERATIONS is None: 
    ITERATIONS = EVALUATIONS // BATCHSIZE + min(EVALUATIONS % BATCHSIZE, 1)

fileprinter = Fileprinter(
    f'Results/History{scenario}.csv',
    20, 
    header = ["objective", "energyloss", "penalties", "Gas GWh p.a.", "Gas CF", 
              "Flex GWh p.a.", "PHES GWh p.a.", "Spillage GWh p.a.", "Trans GWh p.a.",
              "FQ", "NQ", "NS", "NV", "AS", "SW", "TV"] + 
    [f"pv{n}" for n, _ in enumerate(PVl)] + [f"w{n}" for n, _ in enumerate(OnsWl)] +
    [f"gas{n}" for n, _ in enumerate(Nodel)] + [f"php{n}" for n, _ in enumerate(Nodel)] + ["phes"],
    resume = not INIT, 
    )

#%%

@njit
def generate_samples(x, n, width, rng):
    # In place to avoid assigning memory
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = rng.uniform()
    return x

@njit 
def unnormalise(arr, lb, ub):
    return arr * (ub - lb) + lb

if __name__ == "__main__":
    
    if INIT: 
        x = np.stack((ub, (ub+lb)/2, lb)).T
        ObjectiveWrapper(x, costs, fileprinter)
    
    n_inputs = len(lb)
    x = np.empty((BATCHSIZE, n_inputs), np.float64)
    
    rng = np.random.default_rng(RANDOMSEED)
    for _ in tqdm(range(ITERATIONS)):
        x = unnormalise(x, lb, ub)
        ObjectiveWrapper(x.T, costs, fileprinter)
    fileprinter.Terminate()
    
    
    
