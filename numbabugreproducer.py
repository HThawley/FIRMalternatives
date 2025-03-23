# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:22:35 2025

@author: u6942852
"""


from numba import njit, objmode, prange
import numpy as np

@njit
def _add(a):
    with objmode():
        globals()['acc'] += a

@njit(parallel=True)
def parallel_sum_global(arr):
    for i in prange(len(arr)):
        _add(arr[i])

@njit(parallel=False)
def sum_global(arr):
    for i in prange(len(arr)):
        _add(arr[i])

@njit(parallel=True)
def parallel_sum_local(arr):
    acc = 0
    for i in prange(len(arr)):
        acc += arr[i]
    return acc

n = 100
print('True answer:', np.arange(n).sum()) # True answer: 4950 

acc = 0
parallel_sum_global(np.arange(n)) 
print('Numba parallel global answer:', acc) # Numba parallel answer: 78

acc = 0
sum_global(np.arange(n)) 
print('Numba global answer:', acc) # Numba global answer: 4950

acc = parallel_sum_local(np.arange(n))
print('Numba parallel local answer:', acc) # Numba parallel local answer: 4950