# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Reliability(solution, flexible):
    """Single-solution version of Reliability"""

    Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
               solution.GBaseload.sum(axis=1) - flexible)
    intervals = solution.intervals

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    efficiency, resolution = solution.efficiency, solution.resolution 

    solution.Discharge = np.zeros(intervals)
    solution.Charge = np.zeros(intervals)
    solution.Storage = np.zeros(intervals)
    
    solution.Storage[-1] =  0.5*Scapacity
    for t in range(intervals):
        solution.Discharge[t] = np.minimum(np.minimum(np.maximum(0, Netload[t]), Pcapacity), solution.Storage[t-1] / resolution)
        solution.Charge[t] = np.minimum(np.minimum(-1 * np.minimum(0, Netload[t]), Pcapacity), (Scapacity - solution.Storage[t-1]) / efficiency / resolution)
        solution.Storage[t] = solution.Storage[t-1] - solution.Discharge[t] * resolution + solution.Charge[t] * resolution * efficiency

    solution.Deficit = np.maximum(Netload - solution.Discharge, 0)
    solution.Spillage = -1 * np.minimum(Netload + solution.Charge, 0)

    solution.flexible = flexible

    return solution.Deficit

