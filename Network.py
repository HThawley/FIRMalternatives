# A transmission network model to calculate inter-regional power flows
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit

@njit()
def Transmission(solution):
    solution.MHydro = np.atleast_2d(solution.GHydro).T * solution.CHydro / solution.CHydro.sum()
    solution.MBio   = np.atleast_2d(solution.GBio).T   * solution.CBio   / solution.CBio.sum()
    solution.MGas   = np.atleast_2d(solution.GGas).T   * solution.CGas   / solution.CGas.sum()
    
    solution.MDeficit = np.atleast_2d(solution.GDeficit / solution.MLoad.sum(axis=1)).T * solution.MLoad 
    solution.MSpillage = np.atleast_2d(solution.GSpillage / (solution.MPV.sum(axis=1) + solution.MOnsW.sum(axis=1))).T * (solution.MPV+solution.MOnsW)

# =============================================================================
#     seems to handle divide by zero ok - but leaving above code for later dev
#     dzsm = solution.CPHP != 0 # divide by zero safe mask
#     pcfactor = np.zeros(solution.CPHP.shape)
#     pcfactor[dzsm] =  solution.CPHP[dzsm] / solution.CPHP[dzsm].sum(axis=0)
# =============================================================================
    pcfactor = np.atleast_2d(solution.CPHP / solution.GCPHP).T
    solution.MDischarge = (solution.GDischarge * pcfactor).T
    solution.MCharge = (solution.GCharge * pcfactor).T
    solution.MStorage = (solution.GStorage * pcfactor).T

    solution.MImport = (solution.MLoad + solution.MCharge + solution.MSpillage 
                        - solution.MPV - solution.MOnsW - solution.MHydro - 
                        solution.MBio - solution.MGas - solution.MDischarge - 
                        solution.MDeficit - np.atleast_2d(solution.CBaseload)).T 

    solution.TDC = np.zeros((7, solution.intervals), np.float64)
    solution.TDC[0] = - solution.MImport[np.where(solution.Nodel_int==0)[0][0]] if 0 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[4] = - solution.MImport[np.where(solution.Nodel_int==2)[0][0]] if 2 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[5] =   solution.MImport[np.where(solution.Nodel_int==7)[0][0]] if 7 in solution.Nodel_int else np.zeros(solution.intervals, dtype=np.float64)
    solution.TDC[6] = - solution.MImport[np.where(solution.Nodel_int==5)[0][0]]
    solution.TDC[1] =   solution.MImport[np.where(solution.Nodel_int==3)[0][0]] - solution.TDC[0]
    solution.TDC[3] =   solution.MImport[np.where(solution.Nodel_int==6)[0][0]] - solution.TDC[6]
    solution.TDC[2] = - solution.MImport[np.where(solution.Nodel_int==1)[0][0]] - solution.TDC[1] - solution.TDC[3]
    solution.TDC = solution.TDC.T
    return solution.TDC