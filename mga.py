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
def Obj(x):
    S = Solution(x)
    S._evaluate()
    result = np.array([S.LCOE + S.Penalties, 
                       S.LCOE, S.LCOG, S.LCOBS, 
                       S.LCOBT, S.LCOBL], dtype=np.float64)
    return result
    
    
if __name__ == '__main__':
    starttime = dt.datetime.now()
    print("Optimisation starts at", starttime)

    
    z = (pzones+wzones+nodes)
    first_pass = np.array([10.1]*z + [500.1])
    ultralow_res = np.array([1.1]*z + [100]) # 1 GW, 100 GWh
    low_res = np.array([0.1]*z + [10.0]) # 100 MW, 10 GWh
    medium_res = np.array([0.01]*z + [1.0]) # 10 MW, 1 GWh
    high_res = np.array([0.001]*z + [0.01]) # 1 MW, 100 MWh
    ultrahigh_res = np.array([0.000_1]*z + [0.01]) # 0.1 MW, 10 MWh
    polishing = np.array([0.000_001]*z + [0.000_1]) # 1 kW, 100 kWh
    
    res = [first_pass, ultralow_res, low_res, medium_res, high_res, ultrahigh_res, polishing]

    problem = Spacepartition(
        func=Obj, 
        bounds=(lb, ub),
        f_args= (),
        printfile='Results/History{}'.format(scenario) if args.cb == 2 else '',
        vectorizable=False,
        max_dims= -1,
        disp = bool(args.ver),
        restart='Results/backup/History{}'.format(scenario) if args.resume == 1 else '',
        near_optimal=1.05,
        nextras = 0,
        )

    problem.Initiate()
    # raise KeyboardInterrupt
    problem.Step({'max_iter':25,
                  'max_res':res[0],
                  'near_optimal':np.inf, 
                  'max_pop':25,
                  })
    print('step2')
    problem.Step({'max_iter':10,
                  'max_res':res[1],
                  'near_optimal':2.5, 
                  'max_pop':25,
                  })
    print('step3')
    problem.Step({'max_iter':20,
                  'max_res':res[2],
                  'near_optimal':2.5, 
                  'max_pop':25,
                  })
    print('step4')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[3],
                  'near_optimal':1.05, 
                  'max_pop':1000,
                  })
    print('step5')
    problem.Step({'max_iter':20,
                  'max_res':res[3],
                  'near_optimal':1.1, 
                  'max_pop':50,
                  })
    print('step6')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[3],
                  'near_optimal':1.05, 
                  'max_pop':1000,
                  })
    print('step7')
    problem.Step({'max_iter':np.inf,
                  'max_res':res[3]/2,
                  'near_optimal':1.02, 
                  'max_pop':1000,
                  })
    
    print('polish')
    problem._parse_step_dict({'max_res': res[4], 
                    'near_optimal':1.05})
    problem.Polish({'max_res':res[4], 
                    'near_optimal':1.05})
    
    result = problem.ReturnElite()


    endtime = dt.datetime.now()
    print("Optimisation took", endtime - starttime)

    print(result.x, result.f)
