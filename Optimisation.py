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

from Timekeeper import PrintTimekeeper
from Input import *

class FilePrinter:
    def __init__(self, file_name:str, save_freq:int):
        self.file_name=file_name
        self.temp_file_path = '-temp.'.join(self.file_name.split('.'))
        self.save_freq=save_freq
        self.callno = 0
        self.array = None
        
    @keeptime('Manage file print')
    def __call__(self, arr):
        self.callno+=1     
        if self.array is None:
            self.array=arr
        else: 
            self.array = np.concatenate((self.array, arr), axis=0)
        if self.callno % self.save_freq == 0:
            self._flush()
    
    @keeptime('Print to file')
    def _print(self):
        with open(self.temp_file_path, 'a', newline='') as file:
            writer(file).writerows(self.array) 
            file.close()
    
    @keeptime('Copying files')
    def _copyfile(self, forward=True):
        if forward is True:
            try:
                shutil.copyfile(self.file_name, self.temp_file_path)
            except FileNotFoundError as e:
                if self.callno == self.save_freq:
                    pass
                else: 
                    raise e 
                    
        else:
           shutil.copyfile(self.temp_file_path, self.file_name)
           os.remove(self.temp_file_path)
           
    def _flush(self):
        print('\rWriting out to file. Do not interrupt', end='\r')
        self._copyfile(True)
        self._print()
        self._copyfile(False)
        print('\r'+' '*40, end='\r')
        self.array=None
    
    def Terminate(self):
        if self.array is not None:
            self._flush()

@keeptime('Objective')
def ObjectiveWrapper(xs, costs, fileprinter):
    result = ObjectiveParallel(xs.T, costs)
    fileprinter(result[:, 1:]) 
    return result[:, 0]

@keeptime('ObjectiveParallel')
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
    
@keeptime('Optimiser')
def Optimise(costs, init='latinhypercube', x0=None, callback_args=()):
    # print(args.i, args.ml, args.mu, args.p)
    
    starttime = dt.now()
    print("Optimisation starts at", starttime)
    
    fileprinter = FilePrinter(f'Results/History{scenario}.csv', 1)
    
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
    
    PrintTimekeeper()
    raise KeyboardInterrupt
    
    with open('Results/Optimisation_resultx{}.csv'.format(scenario), 'w', newline='') as csvfile:
        writer(csvfile).writerow(result.x)
    
    
    from Dispatch import Analysis
    Analysis(result.x)


