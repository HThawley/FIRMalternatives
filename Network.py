# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution):
    
    if solution.scenario < 20:
        solution.CAC = np.zeros(solution.ninter, np.float64)
        solution.TAC = np.zeros((1,solution.ninter), np.float64)
        return solution.TAC
        
    solution.MPeak = np.atleast_2d(solution.GFlexible).T * solution.CPeak / solution.CPeak.sum()
    solution.MDeficit = np.atleast_2d(solution.GDeficit / solution.MLoad.sum(axis=1)).T * solution.MLoad 
    
    MPW = solution.MPV + solution.MOnsW
    solution.MSpillage = np.atleast_2d(solution.GSpillage / MPW.sum(axis=1)).T * MPW
    
    # dzsm = solution.CPHP != 0 # divide by zero safe mask
    # pcfactor = np.zeros(solution.CPHP.shape)
    # pcfactor[dzsm] =  solution.CPHP[dzsm] / solution.CPHP[dzsm].sum(axis=0)
    
    # seems to handle divide by zero ok - but leaving above code for later dev
    pcfactor =  np.atleast_2d(solution.CPHP / solution.CPHP.sum(axis=0)).T
    
    solution.MDischarge = (solution.GDischarge * pcfactor).T
    solution.MCharge = (solution.GCharge * pcfactor).T
    solution.MStorage = (solution.GStorage * pcfactor).T

    MImport = (solution.MLoad + solution.MCharge + solution.MSpillage \
              - MPW - solution.MBaseload - solution.MPeak - solution.MDischarge - solution.MDeficit).T

    solution.TAC = np.zeros((solution.ninter, solution.intervals), np.float64)
    solution.TAC[0] = - MImport[np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TAC[4] = - MImport[np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TAC[5] =   MImport[np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TAC[6] = - MImport[np.where(solution.Nodel_int==5)[0][0]]
    solution.TAC[1] =   MImport[np.where(solution.Nodel_int==3)[0][0]] - solution.TAC[0]
    solution.TAC[3] =   MImport[np.where(solution.Nodel_int==6)[0][0]] - solution.TAC[6]
    solution.TAC[2] = - MImport[np.where(solution.Nodel_int==1)[0][0]] - solution.TAC[1] - solution.TAC[3]
    solution.TAC = solution.TAC.T
    
    solution.CAC = np.zeros(solution.ninter, dtype=np.float64)
    for j in range(solution.ninter):
        for i in range(len(solution.TAC)):
            solution.CAC[j] = np.maximum(abs(solution.TAC[i, j]), solution.CAC[j])
    
    return solution.TAC
