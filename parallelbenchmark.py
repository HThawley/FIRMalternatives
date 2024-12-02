# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:53:24 2024

@author: u6942852
"""

#benchmarkparallel methods


from numba import njit, prange
from multiprocessing import Pool
import numpy as np
from time import perf_counter
from psutil import cpu_count

from Input import *
from mga import Obj

ncpus = cpu_count(logical=True)


ncases = 1000
cases = (np.random.rand(ncases, len(lb))*(ub-lb) + lb )
nextras=5

#compile
Obj(cases[0])

def serial():
    start = perf_counter()
    result = np.array([Obj(x) for x in cases])
    end = perf_counter()
    print(f"Serial evaluation took {end-start} seconds.")
    return result

def mpObjWrapper(x):
    return Obj(x)   
  
def multiprocessed():
 
    with Pool(processes=min(ncases, ncpus)) as processPool:
        result = processPool.imap(mpObjWrapper, [x for x in cases[:ncpus]], chunksize=1)
        result = np.array([res for res in result])
        
        start = perf_counter()
        result = processPool.imap(mpObjWrapper, [x for x in cases], chunksize=ncases//ncpus + 1)
        result = np.array([res for res in result])
        
        processPool.terminate()
    end = perf_counter()
    print(f"Multiprocessing evaluation took {end-start} seconds.")

    return result

@njit(parallel=True)
def jitObjWrapper(cases_):
    result = np.empty((ncases, 1+nextras))
    for i in prange(ncases):
        result[i] = Obj(cases_[i])
    return result

def jitted():
    jitObjWrapper(cases[:2])#compile
    start=perf_counter()
    result = jitObjWrapper(cases)
    end=perf_counter()
    print(f"numba evaluation took {end-start} seconds.")
    return result




if __name__=='__main__':
    
    serial()
    # mpObjWrapper(cases[0])
    multiprocessed()
    jitted()
    