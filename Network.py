# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution):
    
    solution.MPeak = np.atleast_2d(solution.flexible).T * solution.CPeak / solution.CPeak.sum()
    solution.MDeficit = np.atleast_2d(solution.Deficit / solution.MLoad.sum(axis=1)).T * solution.MLoad 
    
    solution.MPV, solution.MOnsW, MPW = solution.GPV, solution.GOnsW, solution.GPV + solution.GOnsW
    solution.MSpillage = np.atleast_2d(solution.Spillage / MPW.sum(axis=1)).T * MPW
    
    # dzsm = solution.CPHP != 0 # divide by zero safe mask
    # pcfactor = np.zeros(solution.CPHP.shape)
    # pcfactor[dzsm] =  solution.CPHP[dzsm] / solution.CPHP[dzsm].sum(axis=0)
    
    # seems to handle divide by zero ok - but leaving above code for later dev
    pcfactor =  np.atleast_2d(solution.CPHP / solution.CPHP.sum(axis=0)).T
    
    solution.MDischarge = (solution.Discharge * pcfactor).T
    solution.MCharge = (solution.Charge * pcfactor).T
    solution.MStorage = (solution.Storage * pcfactor).T

    MImport = (solution.MLoad + solution.MCharge + solution.MSpillage \
              - MPW - solution.GBaseload - solution.MPeak - solution.MDischarge - solution.MDeficit).T

    solution.TDC = np.zeros((7, solution.intervals), np.float64)
    solution.TDC[0] = - MImport[np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[4] = - MImport[np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[5] = MImport[np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[6] = - MImport[np.where(solution.Nodel_int==5)[0][0]]
    solution.TDC[1] = MImport[np.where(solution.Nodel_int==3)[0][0]] - solution.TDC[0]
    solution.TDC[3] = MImport[np.where(solution.Nodel_int==6)[0][0]] - solution.TDC[6]
    solution.TDC[2] = - MImport[np.where(solution.Nodel_int==1)[0][0]] - solution.TDC[1] - solution.TDC[3]
    solution.TDC = solution.TDC.T
    return solution.TDC
