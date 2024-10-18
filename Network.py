# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution, output=False):
    
    solution.MPeak = np.atleast_2d(solution.flexible).T * np.atleast_2d(solution.CPeak / solution.CPeak.sum())
    solution.MDeficit = np.atleast_2d(solution.Deficit).T * np.divide(solution.MLoad, solution.MLoad.sum(axis=1).reshape(-1, 1))

    MPW = solution.GPV + solution.GWind
    solution.MSpillage = np.atleast_2d(solution.Spillage).T * np.divide(MPW, np.atleast_2d(MPW.sum(axis=1) + 0.00000001).T) 
    solution.MPV, solution.MWind = solution.GPV, solution.GWind
    
    # dzsm = solution.CPHP != 0 # divide by zero safe mask
    # pcfactor = np.zeros(solution.CPHP.shape)
    # pcfactor[dzsm] =  solution.CPHP[dzsm] / solution.CPHP[dzsm].sum(axis=0)
    
    # seems to handle divide by zero ok - but leaving above code for later dev
    pcfactor =  solution.CPHP / solution.CPHP.sum(axis=0)
    
    solution.MDischarge = (np.atleast_2d(solution.Discharge).T * pcfactor)# MDischarge: DPH(j, t)
    solution.MCharge = (np.atleast_2d(solution.Charge).T * pcfactor) # MCharge: CHPH(j, t)
    solution.MStorage = np.atleast_2d(solution.Storage).T * pcfactor # SPH(t, j), MWh

    MImport = solution.MLoad + solution.MCharge + solution.MSpillage \
              - MPW - solution.GBaseload - solution.MPeak - solution.MDischarge - solution.MDeficit # EIM(t, j), MW

    solution.FQ = -1 * MImport[:, np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.AS = -1 * MImport[:, np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.SW = MImport[:, np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TV = -1 * MImport[:, np.where(solution.Nodel_int==5)[0][0]]

    solution.NQ = MImport[:, np.where(solution.Nodel_int==3)[0][0]] - solution.FQ
    solution.NV = MImport[:, np.where(solution.Nodel_int==6)[0][0]] - solution.TV

    solution.NS = -1. * MImport[:, np.where(solution.Nodel_int==1)[0][0]] - solution.NQ - solution.NV

    return np.stack((solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV), axis=1)
