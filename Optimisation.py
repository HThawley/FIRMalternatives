# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import os
import shutil
import numpy as np
from csv import writer
from numba import njit,prange
from scipy.optimize import differential_evolution
from scipy._lib._util import check_random_state
from datetime import datetime as dt

from Input import *

def ObjectiveWrapper(xs, costs):
    result = ObjectiveParallel(xs.T, costs)
    print('\rWriting out to file. Do not interrupt', end='\r')
    path, temppath = f'Results/History{scenario}.csv', f'Results/History{scenario}-temp.csv'
    shutil.copyfile(path, temppath)
    with open(temppath, 'a', newline='') as file:
        writer(file).writerows(result[:, 1:]) 
        file.close()
    print('\r'+' '*40, end='\r')
    shutil.copyfile(temppath, path)
    os.remove(temppath)
    
    return result[:, 0]

@njit(parallel=True)
def ObjectiveParallel(xs, costs):
    result = np.empty((len(xs), 15), dtype=np.float64)
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
        (S.GHydro.sum() + S.GBio.sum() + S.CBaseload.sum()*S.intervals
         ) * S.resolution / S.years, # Flexible GWh p.a.
        S.GDischarge.sum() * S.resolution / S.years, # PHES GWh p.a.
        S.GSpillage.sum() * S.resolution / S.years, # Spillage GWh p.a.
        np.abs(S.TDC).sum() * S.resolution / S.years, # Transmission GWh p.a.
        ] + list(S.CDC) 
        )
        
class CallbackClass:
    def __init__(self, step=50, conv=100):
        """
        This object is called after each iteration.
        Step - how often (# iterations) to print intermediate results to console
        Conv - # of iterations with no improvement to best objective after which to terminate
        """
        self.it = 0 
        self.step = step
        self.conv = conv//step
        self.conv_counter = 0
        self.start = dt.now()
        self.elite = np.inf
    def __call__(self, intermediate_result):
        if self.it % self.step == 0:
            print(f'Iteration: {self.it}. Time taken: {dt.now()-self.start}. Best value: {intermediate_result.fun}')
        if intermediate_result.fun == self.elite:
            self.conv_counter+=1
        if self.conv_counter == self.conv:
            return True
        if intermediate_result.fun < self.elite:
            self.elite = intermediate_result.fun
            self.conv_counter=0
        self.it+=1
        return False
    

# class Strategy:
#     def __init__(self):
#         pass
#     def __call__(self, candidate:int, population:np.ndarray, rng=None) -> np.ndarray:
#         obj = ObjectiveParallel(population, costs)[:,0]
#         scale = rng.uniform(args.ml, args.mu)
        
        
        
#         self.candidate = candidate
#         self.population = population
#         raise Exception
#         return population[candidate]

def Optimise(init='latinhypercube', x0=None):
    print(args.i, args.ml, args.mu, args.p)
    starttime = dt.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=ObjectiveWrapper, 
        args=(costs,),
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=(args.ml, args.mu), 
        recombination=args.r,
        disp=True, 
        polish=False, 
        updating='deferred', 
        vectorized=True,
        strategy='currenttobest1bin',#Strategy(),
        init=init,
        x0=x0,
        # callback=CallbackClass(25, 50)
        # workers=1, #vectorisation overrides mp
        )
    
    endtime = dt.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken

if __name__=='__main__':
    raise KeyboardInterrupt
    result, time = Optimise()
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer(csvfile).writerow(result.x)
    
    
    from Dispatch import Analysis
    Analysis(result.x)


