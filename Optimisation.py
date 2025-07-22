# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from csv import writer
from numba import njit,prange
from scipy.optimize import differential_evolution
# from scipy._lib._util import check_random_state
from datetime import datetime as dt

from Input import *
from Timekeeper import PrintTimekeeper, keeptime, timekeeper
from Fileprinter import Fileprinter

# @keeptime('Objective', TK_SWITCH)
def ObjectiveWrapper(xs, costs, fileprinter):
    result = ObjectiveParallel(xs.T, costs)
    fileprinter(result[:, 1:]) 
    return result[:, 0]

# @keeptime('ObjectiveParallel', TK_SWITCH)
@njit(parallel=True)
def ObjectiveParallel(xs, costs):
    result = np.empty((len(xs), 16), dtype=np.float64)
    for i in prange(len(xs)):
        result[i, :] = Objective(xs[i], costs)
    result = np.concatenate((result, xs), axis=1)
    return result

@njit
def Objective(x, costs):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate(costs)
    return np.array([
        S.LCOE + S.Penalties, # objective
        S.energyloss, # Energy served - transmission loss
        S.Penalties, # penalties
        S.GGas.sum() * S.resolution/S.years, # Gas GWh p.a.
        100*S.GGas.sum() * S.resolution/S.years / (S.CGas.sum()*8760), # Gas CF
        (S.GHydro.sum() + S.GBio.sum() + S.CBaseload.sum()*S.intervals
         ) * S.resolution / S.years, # Flexible GWh p.a.
        S.GDischarge.sum() * S.resolution / S.years, # PHES GWh p.a.
        S.GSpillage.sum() * S.resolution / S.years, # Spillage GWh p.a.
        np.abs(S.TDC).sum() * S.resolution / S.years, # Transmission GWh p.a.
        ] + list(S.CDC) 
        )
        
class CallbackClass:
    def __init__(self, display=50, stagnation=100, stag_rate=1e-6):
        """
        This object is called after each iteration.
        display    - how often (# iterations) to print intermediate results to console
        stagnation - # of iterations with no improvement to best objective after which to terminate
        """
        self.it = 0 
        self.display = display
        self.stagnation = stagnation
        self.stag_counter = 0
        self.stag_rate = stag_rate
        self.start = dt.now()
        self.elite = np.inf
    def __call__(self, intermediate_result):
        if self.it % self.display == 0:
            print(f'Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}')
        if self.elite - intermediate_result.fun < self.stag_rate:
            self.stag_counter+=1
        else: 
            self.elite = intermediate_result.fun
            self.stag_counter=0
        if self.stag_counter == self.stagnation:
            print(f'Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}')
            return True
        
        self.it+=1
        return False
    
# @keeptime('Optimiser', TK_SWITCH)
def Optimise(costs, init='latinhypercube', x0=None, callback_args=()):
    # print(args.i, args.ml, args.mu, args.p)
    
    starttime = dt.now()
    print("Optimisation starts at", starttime)
    
    fileprinter = Fileprinter(f'Results/History{scenario}.csv', 20, resume=bool(args.res))
    
    result = differential_evolution(
        func=ObjectiveWrapper, 
        args=(costs, fileprinter),
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i,
        popsize=args.p, 
        mutation=(args.ml, args.mu), 
        recombination=args.r,
        disp=bool(args.ver), 
        polish=False, 
        updating='deferred', 
        vectorized=True,
        strategy='currenttobest1bin',
        init=init,
        x0=x0,
        callback=CallbackClass(*callback_args)
        )
    
    fileprinter.Terminate()
    endtime = dt.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken

if __name__=='__main__':
    # timekeeper = Timekeeper()
    
    result, time = Optimise(costs)
    
    PrintTimekeeper(f'Results/Timekeep-opt-{scenario}.csv')
    raise KeyboardInterrupt
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer(csvfile).writerow(result.x)
    
    
    from Dispatch import Analysis
    Analysis(result.x)


