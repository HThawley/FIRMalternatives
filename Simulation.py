import numpy as np
from numba import njit

@njit()
def Reliability(solution, flexible):
    solution.GNetload = solution.MLoad.sum(axis=1) - solution.MPV.sum(axis=1) - solution.MOnsW.sum(axis=1) - flexible - solution.CBaseload.sum()

    solution.GDischarge = np.zeros(solution.intervals)
    solution.GCharge = np.zeros(solution.intervals)
    solution.GStorage = np.zeros(solution.intervals)
    solution.GStorage[-1] = 0.5*solution.CPHS
    for t in range(solution.intervals):
        solution.GDischarge[t] = np.minimum(np.minimum(np.maximum(0, solution.GNetload[t]), solution.GCPHP), solution.GStorage[t-1] / solution.resolution)
        solution.GCharge[t] = np.minimum(np.minimum(-1 * np.minimum(0, solution.GNetload[t]), solution.GCPHP), (solution.CPHS - solution.GStorage[t-1]) / solution.efficiency / solution.resolution)
        solution.GStorage[t] = solution.GStorage[t-1] - solution.GDischarge[t] * solution.resolution + solution.GCharge[t] * solution.resolution * solution.efficiency

    solution.GDeficit = np.maximum(solution.GNetload - solution.GDischarge, 0)
    solution.GSpillage = - np.minimum(solution.GNetload + solution.GCharge, 0)
    solution.GFlexible = flexible
    
    return solution.GDeficit
