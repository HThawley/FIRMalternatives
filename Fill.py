from numba import njit
import numpy as np
from Simulation import Reliability 

#%%
@njit()
def Fill(solution):
    ### maximum dispatch of gas
    solution.GGas = solution.CGas.sum() * np.ones(solution.intervals, dtype=np.float64)
    Reliability(solution, flexible=solution.GGas)
    solution.GGas = np.maximum(solution.GGas-solution.GSpillage, 0)
    
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
        if fill > 0:
            # simplified model of trickle-charging constraints
            # TODO: consider adjusting energy constraint with previous fillage to reduce spillage
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_trickle[t] = flex
    
    Reliability(solution, flexible=flex_power+flex_trickle+solution.GGas)
    flex_trickle = np.maximum(flex_trickle-solution.GSpillage, 0)

    ### enforce annual energy constraint
    bio_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Bio_res  
    if bio_exceedance > 0:
        ## Remove the lowest power instances of Bio use until energy exceedance is eliminated
        bio_sorted = np.argsort(flex_trickle)
        flex_trickle[bio_sorted[:np.where(flex_trickle[bio_sorted].cumsum()>bio_exceedance)[0][1]]]=0
        bio_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Bio_res
        if bio_exceedance > 0:
            bio_sorted = np.argsort(flex_power)
            flex_power[bio_sorted[:np.where(flex_power[bio_sorted].cumsum()>bio_exceedance)[0][1]]]=0
            bio_exceedance=0
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
        if fill > 0:
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_trickle[t] = flex
            
    Reliability(solution, flexible=flex_power+flex_trickle+solution.GBio+solution.GGas)
    flex_trickle = np.maximum(flex_trickle-solution.GSpillage, 0)
    
    ### enforce annual energy constraint
    hydro_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Hydro_res
    if hydro_exceedance > 0:
        ## Remove the lowest power instances of Hydro use until energy exceedance is eliminated
        hydro_sorted = np.argsort(flex_trickle)
        flex_trickle[hydro_sorted[:np.where(flex_trickle[hydro_sorted].cumsum()>hydro_exceedance)[0][1]]]=0
        hydro_exceedance = flex_power.sum() + flex_trickle.sum() - solution.Hydro_res
        if hydro_exceedance > 0:
            hydro_sorted = np.argsort(flex_power)
            flex_power[hydro_sorted[:np.where(flex_power[hydro_sorted].cumsum()>hydro_exceedance)[0][1]]]=0
            hydro_exceedance=0
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
        if fill > 0:
            fill = min(fill, (solution.CPHS - solution.GStorage[t-1])/solution.resolution/solution.efficiency)
            flex = min(fill, flex_cap - flex_power[t], solution.GCPHP - solution.GCharge[t] + solution.GDischarge[t])
            fill -= flex
            flex_power[t] += flex
    # Reliability(solution, flexible=solution.GHydro+solution.GBio+flex_power)
    # flex_power = np.maximum(flex_power-solution.GSpillage, 0)
    
    ### apportion as much gas usage to hydro and bio as possible
    if hydro_exceedance < -10:
        # This flex_trickle is not being used as it has been previously, but it is a 
        #    convenient spare array to avoid the time cost of allocating new memory
        flex_trickle = np.minimum(flex_power, solution.CHydro.sum()-solution.GHydro)
        flex_trickle = flex_trickle * min(1, -hydro_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
        solution.GHydro += flex_trickle
        flex_power -= flex_trickle
    
    if bio_exceedance < -10:
        # This flex_trickle is not being used as it has been previously, but it is a 
        #    convenient spare array to avoid the time cost of allocating new memory
        flex_trickle = np.minimum(flex_power, solution.CBio.sum()-solution.GBio)
        flex_trickle = flex_trickle * min(1, -bio_exceedance/flex_trickle.sum()) if flex_trickle.sum() > 0 else flex_trickle
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

    