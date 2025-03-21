# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:17:48 2024

@author: u6942852
"""


import numpy as np
import datetime as dt
from numba import njit
from csv import writer

from Input import *
from spacepartition import Spacepartition

@njit
def Obj(x, costs):
    S = Solution(x)
    S._evaluate(costs)
    result = np.array([S.LCOE + S.Penalties, 
                       S.LCOE, S.LCOG, S.LCOBS, 
                       S.LCOBT, S.LCOBL], dtype=np.float64)
    return result
    
def mp_Obj_wrapper(x):
    return Obj(x)
    

mp = args.mp
if mp.lower() == 'pool':
    func = mp_Obj_wrapper
elif mp.lower() == 'jit':
    func = Obj
else:
    raise ValueError
    
z = (pzones+wzones+phnodes)
first_pass = np.array([8.1]*z + [256.1])
ultralow_res = np.array([2.1]*z + [64.1]) 
low_res = np.array([1.1]*z + [16.1]) # 100 MW, 10 GWh
medium_res = np.array([0.51]*z + [8.1]) # 10 MW, 1 GWh
high_res = np.array([0.251]*z + [4.1]) # 1 MW, 100 MWh
ultrahigh_res = np.array([0.000_1]*z + [0.01]) # 0.1 MW, 10 MWh
polishing = np.array([0.000_001]*z + [0.000_1]) # 1 kW, 100 kWh

res = [first_pass, ultralow_res, low_res, medium_res, high_res, ultrahigh_res, polishing]

if __name__ == '__main__':

    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    problem = Spacepartition(
        func=func, 
        multiprocessing=mp,
        bounds=(lb, ub),
        f_args=(costs,),
        printfile='Results/History{}'.format(scenario) if args.cb == 2 else '',
        vectorizable=False,
        max_dims=args.d,
        disp = bool(args.ver),
        restart='Results/History{}'.format(scenario) if args.resume == 1 else '',
        nextras = 5,
        )

    problem.Initiate()
    
    print('step 1')
    problem.Step({'max_iter':10,
                  'max_res':res[0],
                  'near_optimal':np.inf, 
                  'max_pop':25,
                  })
    print('step 2')
    problem.Step({'max_iter':10,
                  'max_res':res[1],
                  'near_optimal':2.5, 
                  'max_pop':1000,
                  })
    
    print('step 3')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[1],
                  'near_optimal':1.1, 
                  'max_pop':20000,
                  })
    print('step 4')
    problem.Step({'max_iter':10,
                  'max_res':res[2],
                  'near_optimal':2.5, 
                  'max_pop':1000,
                  })
    print('step 5')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[2],
                  'near_optimal':1.1, 
                  'max_pop':20000,
                  })
    print('step 6')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[3],
                  'near_optimal':1.25, 
                  'max_pop':20000,
                  })
    
    print('step 7')
    problem.Step({'max_iter':20,
                  'max_res':res[3],
                  'near_optimal':2.0, 
                  'max_pop':10000,
                  })
    print('step 8')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[3],
                  'near_optimal':1.2, 
                  'max_pop':20000,
                  })
    print('polish')
    problem.Polish({'max_res':res[3], 
                    'near_optimal':1.15})
    
    result = problem.ReturnElite()


    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result.x, result.f)

    from Fill import Analysis
    Analysis(Solution(result.x))
