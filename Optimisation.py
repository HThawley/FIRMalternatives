# To optimise the configurations of energy generation, storage and transmission assets
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import datetime as dt
import csv
import numpy as np
from numba import njit,prange
from scipy.optimize import differential_evolution

from Input import *



@njit(parallel=True)
def ObjectiveWrapper(xs):
    result = np.empty(xs.shape[1], dtype=np.float64)
    for i in prange(xs.shape[1]):
        result[i] = Objective(xs[:,i])
    return result

@njit
def Objective(x):
    """This is the objective function"""
    S = Solution(x)
    S._evaluate()
    return S.LCOE + S.Penalties

def Callback_1(xk, convergence=None):
    with open('Results/History{}.csv'.format(scenario), 'a', newline='') as csvfile:
        csv.writer(csvfile).writerow([Objective(xk)] + list(xk))
        
def Init_callback():
    with open('Results/History{}.csv'.format(scenario), 'w', newline='') as csvfile:
        csv.writer(csvfile)

def Optimise():
    if args.cb > 1 and args.resume ==0: 
        Init_callback()

    if args.resume == 1:
        x0 = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    else:
        x0 = None
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)
    result = differential_evolution(
        func=ObjectiveWrapper, 
        x0=x0,
        bounds=list(zip(lb, ub)), 
        tol=0,
        maxiter=args.i, 
        popsize=args.p, 
        mutation=args.m, 
        recombination=args.r,
        disp=bool(args.ver), 
        polish=False, 
        updating='deferred', 
        vectorized=True,
        callback=Callback_1 if args.cb == 1 else None,
        workers=1, #vectorisation overrides mp
        )
    
    endtime = dt.datetime.now()
    timetaken = endtime-starttime
    print("Optimisation took", timetaken)

    return result, timetaken

if __name__=='__main__':

    result, time = Optimise()
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(result.x)
    
    
    from Fill import Analysis
    Analysis(result.x)


