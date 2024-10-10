import numpy as np 
from numba import njit
import datetime as dt

from Input import * 
from Simulation import Reliability

@njit 
def Fill(solution):
    
    flexible = np.zeros(intervals, dtype=np.float64)
    deficit = Reliability(solution, flexible=flexible)
    
    flex_cap = CPeak.sum()*1000
    
    fill = 0
    for t in range(intervals-1, -1, -1):
        d = deficit[t]
        if d > 0:
            flex = min(d, flex_cap)
            flexible[t] = flex
    
            fill += (d-flex)/efficiency

        if fill > 0:
            flex = min(fill, flex_cap - flexible[t])
            fill -= flex
            flexible[t] += flex
    Deficit = Reliability(solution, flexible=flexible)
    return flexible

def Analysis(x):
    """Dispatch.Analysis(result.x)"""

    starttime = dt.datetime.now()
    print('Fill starts at', starttime)

    Flex = Fill(Solution(x))
    np.savetxt('Results/Dispatch_Flexible{}.csv'.format(scenario), Flex, fmt='%f', delimiter=',', newline='\n', header='Flexible energy resources')

    endtime = dt.datetime.now()
    print('Fill took', endtime - starttime)

    from Statistics import Information
    Information(x, Flex)

    return True

if __name__ == '__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    
    Analysis(x)