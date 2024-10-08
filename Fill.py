from Input import * 
from Simulation import Reliability 

import numpy as np 

@njit()
def Fill(solution):
    
    flexible = np.zeros(intervals, dtype=np.float64)
    deficit = Reliability(solution, flexible=flexible)
    flex_cap = CPeak.sum()*1000
    
    fill = 0    
    for t in range(intervals-1, -1, -1):
        d = deficit[t]
        if d > 0:
            flex = min(d, flex_cap - flexible[t]) 
            flexible[t] = flex
            if d-flex > 0:
                fill += (d-flex)/efficiency
        if fill > 0:
            flex = min(fill, flex_cap - flexible[t]) 
            fill -= flex
            flexible[t] += flex
    
    Deficit = Reliability(solution, flexible=flexible)
    return Deficit
            
if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
   
    solution = Solution(x)
    deficit = Reliability(solution, flexible=np.zeros(intervals, dtype=np.float64))
    Fill(solution)
    
