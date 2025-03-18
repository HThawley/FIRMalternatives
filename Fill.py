import numpy as np 
from numba import njit
import datetime as dt

# from Input import * 
from Simulation import Reliability

@njit 
def Fill(solution):
    flexible = np.zeros(solution.intervals, dtype=np.float64)
    Reliability(solution, flexible=flexible)
    
    fill = 0
    for t in range(solution.intervals-1, -1, -1):
        if solution.GDeficit[t] > 0:
            flexible[t] = min(solution.GDeficit[t], solution.GCPeak)
            fill += (solution.GDeficit[t]-flexible[t])/solution.efficiency

        if fill > 0:
            # simplified charging model
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, solution.GCPeak - flexible[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])

            fill -= flex
            flexible[t] += flex
    Reliability(solution, flexible=flexible)
    return flexible


def Analysis(solution):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Fill starts at', starttime)
    Flex = Fill(solution)
    endtime = dt.datetime.now()
    print('Fill took', endtime - starttime)

    np.savetxt(f'Results/Dispatch_Flexible{solution.scenario}.csv', Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    from Statistics import Information
    Information(solution.x)

    return True

if __name__ == '__main__':
    from Input import * 
    x = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',', dtype=float)
    
    Analysis(Solution(x))