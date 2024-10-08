# To simulate energy supply-demand balance based on long-term, high-resolution chronological data
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Reliability(solution, flexible, start=None, end=None):
    """Single-solution version of Reliability"""
    assert solution.nvec == 1 
    assert solution.vectorised is False

    if start is None and end is None: 
        Netload = (solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
                   solution.GBaseload.sum(axis=1) - flexible)
        intervals = solution.intervals

    else: 
        Netload = ((solution.MLoad.sum(axis=1) - solution.GPV.sum(axis=1) - solution.GWind.sum(axis=1) -
                   solution.GBaseload.sum(axis=1))[start:end] - flexible)
        intervals = len(Netload)

    Pcapacity = solution.CPHP.sum() * 1000 # S-CPHP(j), GW to MW
    Scapacity = solution.CPHS * 1000 # S-CPHS(j), GWh to MWh
    efficiency, resolution = solution.efficiency, solution.resolution 

    Discharge = np.zeros(intervals)
    Charge = np.zeros(intervals)
    Storage = np.zeros(intervals)
    
    Storaget_1 =  0.5*Scapacity
    for t in range(intervals):
        Netloadt = Netload[t]
        

        Discharget = np.minimum(np.minimum(np.maximum(0, Netloadt), Pcapacity), Storaget_1 / resolution)
        Charget = np.minimum(np.minimum(-1 * np.minimum(0, Netloadt), Pcapacity), (Scapacity - Storaget_1) / efficiency / resolution)
        Storaget = Storaget_1 - Discharget * resolution + Charget * resolution * efficiency
        Storaget_1 = Storaget
        
        Discharge[t] = Discharget
        Charge[t] = Charget
        Storage[t] = Storaget

    Deficit = np.maximum(Netload - Discharge, np.zeros(intervals))
    Spillage = -1 * np.minimum(Netload + Charge, np.zeros(intervals))

    solution.flexible = flexible
    solution.Spillage = Spillage
    solution.Charge = Charge
    solution.Discharge = Discharge
    solution.Storage = Storage
    solution.Deficit = Deficit

    return Deficit

