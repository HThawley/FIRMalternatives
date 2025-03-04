from numba import njit
import numpy as np
from Simulation import Reliability 


#%%
@njit()
def Fill(solution):
    ### maximum dispatch of gas
    solution.GGas = solution.CGas.sum() * np.ones(solution.intervals, dtype=np.float64)
    Reliability(solution, flexible=solution.GGas)
    
    ### dispatch Bio to fill the gaps
    flex_power   = np.zeros(solution.intervals, np.float64)
    flex_trickle = np.zeros(solution.intervals, np.float64)

    fill, flex_cap = 0, solution.CBio.sum()

    for t in range(solution.intervals-1, -1, -1):
        d = solution.GDeficit[t]
        if d > 0:
            flex = min(d, flex_cap) 
            flex_power[t] = flex
            if d-flex > 0:
                fill += (d-flex)/solution.efficiency
        if fill > 0 and flex_power[t] < flex_cap:
            # simplified model of trickle-charging constraints
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_trickle[t] = flex

    ### enforce annual energy constraint
    flex_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Bio_res  
    if flex_exceedance > 0:
        flex_trickle = flex_trickle*max(0, 1-flex_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
        flex_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Bio_res
        if flex_exceedance > 0:
            ## Remove the lowest power instances of Hydro use until energy exceedance is eliminated
            flex_sorted = np.argsort(flex_power)
            flex_power[flex_sorted[:np.where(flex_power[flex_sorted].cumsum()>flex_exceedance)[0][1]]]=0
            flex_exceedance=0
    solution.GBio   = flex_power + flex_trickle
    
    ### Dispatch Hydro to fill gaps
    Reliability(solution, flexible=solution.GBio+solution.GGas)
    flex_power   = np.zeros(solution.intervals, np.float64)
    flex_trickle = np.zeros(solution.intervals, np.float64)
    
    fill, flex_cap = 0, solution.CHydro.sum()
    for t in range(solution.intervals-1, -1, -1):
        d = solution.GDeficit[t]
        if d > 0:
            flex = min(d, flex_cap) 
            flex_power[t] = flex
            if d-flex > 0:
                fill += (d-flex)/solution.efficiency
        if fill > 0 and flex_power[t] < flex_cap:
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_trickle[t] = flex
    
    ## This trim of flex does not make much difference on result
    # Reliability(solution, flexible=flex_power+flex_trickle+solution.GBio+solution.GGas)
    # flex_trickle = np.maximum(flex_trickle-solution.GSpillage, 0)
    
    ### enforce annual energy constraint
    flex_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Hydro_res
    if flex_exceedance > 0:
        flex_trickle = flex_trickle*max(0, 1-flex_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
        flex_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Hydro_res
        if flex_exceedance > 0:
            ## Remove the lowest power instances of Hydro use until energy exceedance is eliminated
            flex_sorted = np.argsort(flex_power)
            flex_power[flex_sorted[:np.where(flex_power[flex_sorted].cumsum()>flex_exceedance)[0][1]]]=0
            flex_exceedance=0
    solution.GHydro = flex_power + flex_trickle
    
    ### reduce Gas usage to the minimum necessary amount
    Reliability(solution, flexible=solution.GBio+solution.GHydro)
    flex_power = np.zeros(solution.intervals, np.float64)
    fill, flex_cap = 0, solution.CGas.sum()
    for t in range(solution.intervals-1, -1, -1):
        d = solution.GDeficit[t]
        if d > 0:
            flex = min(d, flex_cap) 
            flex_power[t] = flex
            if d-flex > 0:
                fill += (d-flex)/solution.efficiency
        if fill > 0 and flex_power[t] < flex_cap:
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_power[t] += flex
    Reliability(solution, flexible=solution.GHydro+solution.GBio+flex_power)
    flex_power = np.maximum(flex_power-solution.GSpillage, 0)
    
    ### apportion as much gas usage to hydro and bio as possible
    flex_exceedance = solution.GHydro.sum() - solution.Hydro_res
    if flex_exceedance < -10:
        # This flex_trickle is not being used as it has been previously, but it is a 
        #    convenient spare array to avoid the time cost of allocating new memory
        flex_trickle = np.minimum(flex_power, solution.CHydro.sum()-solution.GHydro)
        flex_trickle = flex_trickle * min(1, -flex_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
        solution.GHydro += flex_trickle
        flex_power -= flex_trickle
    
    flex_exceedance = solution.GBio.sum() - solution.Bio_res
    if flex_exceedance < -10:
        # This flex_trickle is not being used as it has been previously, but it is a 
        #    convenient spare array to avoid the time cost of allocating new memory
        flex_trickle = np.minimum(flex_power, solution.CBio.sum()-solution.GBio)
        flex_trickle = flex_trickle * min(1, -flex_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
        solution.GBio += flex_trickle
        flex_power -= flex_trickle
    
    solution.GGas = flex_power
    
    return Reliability(solution, flexible=solution.GBio+solution.GGas+solution.GHydro)


if __name__=='__main__':
    from Input import * 
    
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
   
    solution = Solution(x)
    solution.CGas = np.repeat(2/5, 5)
    deficit = Fill(solution)

    