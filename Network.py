# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution, output=False):
    
    MPeak = np.atleast_2d(solution.flexible).T * np.atleast_2d(solution.CPeak / solution.CPeak.sum())
    MDeficit = np.atleast_2d(solution.Deficit).T * np.divide(solution.MLoad, solution.MLoad.sum(axis=1).reshape(-1, 1))

    MPW = solution.GPV + solution.GWind
    MSpillage = np.atleast_2d(solution.Spillage).T * np.divide(MPW, np.atleast_2d(MPW.sum(axis=1) + 0.00000001).T) 

    # dzsm = solution.CPHP != 0 # divide by zero safe mask
    # pcfactor = np.zeros(solution.CPHP.shape)
    # pcfactor[dzsm] =  solution.CPHP[dzsm] / solution.CPHP[dzsm].sum(axis=0)
    
    # seems to handle divide by zero ok - but leaving above code for later dev
    pcfactor =  solution.CPHP / solution.CPHP.sum(axis=0)
    
    MDischarge = (np.atleast_2d(solution.Discharge).T * pcfactor)# MDischarge: DPH(j, t)
    MCharge = (np.atleast_2d(solution.Charge).T * pcfactor) # MCharge: CHPH(j, t)

    MImport = solution.MLoad + MCharge + MSpillage \
              - MPW - solution.GBaseload - MPeak - MDischarge - MDeficit # EIM(t, j), MW

    FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    SW = MImport[:, np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0]]

    NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0]] - FQ
    NV = MImport[:, np.where(solution.Nodel_int==6)[0][0]] - TV

    NS = -1. * MImport[:, np.where(solution.Nodel_int==1)[0][0]] - NQ - NV
    
    assert np.abs(NS - MImport[:, np.where(solution.Nodel_int==4)[0][0]] - AS + SW).sum()<=1

    if output is True:
        MStorage = np.atleast_2d(solution.Storage).T * pcfactor # SPH(t, j), MWh
        solution.MPV, solution.MWind, solution.MBaseload, solution.MPeak = (solution.GPV, solution.GWind, solution.GBaseload, MPeak)
        solution.MDischarge, solution.MCharge, solution.MStorage = (MDischarge, MCharge, MStorage)
        solution.MDeficit, solution.MSpillage = (MDeficit, MSpillage)

    return np.stack((FQ, NQ, NS, NV, AS, SW, TV), axis=1)
