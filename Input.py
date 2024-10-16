# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, float64, int64, prange, boolean
from numba.experimental import jitclass

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')

parser.add_argument('-s', default=31, type=int, required=False, help='11, 12, 13, ...')

parser.add_argument('-cb', default=2, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-resume', default=0, type=int, required=False, help='Boolean - whether to restart')

args = parser.parse_args()
scenario = args.s

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
Windl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, Windl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, Windl))
Nodel_int, PVl_int, Windl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, Windl_int))

resolution = 0.5
firstyear, finalyear, timestep = (2020, 2029, 1)

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSWind = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Windl))) # TSWind(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DCloss = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) * 0.03 * pow(10, -3)
CDC6max = 3 * 0.63 # GW

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

if scenario<=17:
    node = Nodel[scenario % 10]

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSWind = TSWind[:, Windl==node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel==node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, Windl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[x==node] for x in (Nodel, PVl, Windl)]

elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    MLoad = MLoad[:, np.in1d(Nodel, coverage)]
    TSPV = TSPV[:, np.in1d(PVl, coverage)]
    TSWind = TSWind[:, np.in1d(Windl, coverage)]
    CHydro, CBio, CBaseload, CPeak = [x[np.in1d(Nodel, coverage)] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, Windl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, Windl)]

if scenario >= 31:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    
    TSPV = np.stack([TSPV[:, PVl==node].mean(axis=1) for node in coverage]).T
    TSWind = np.stack([TSWind[:, Windl==node].mean(axis=1) for node in coverage]).T
    # having full of zeros and setting lb,ub=0,0 makes code faster
    TSPV = np.nan_to_num(TSPV, False, 0)
    warnings.simplefilter('default', RuntimeWarning)
    
    Nodel_int, PVl_int, Windl_int = [np.unique(x) for x in (Nodel_int, PVl_int, Windl_int)]
    Nodel, PVl, Windl = [np.unique(x)  for x in (Nodel, PVl, Windl)]
    
    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (len(PVl), len(Windl))
if scenario >= 31:
    pzones+=1
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array([0., 0., 0., 0., 0.] + [0.]   * wzones + contingency   + [0.])
ub = np.array([32., 32., 32., 0, 32.] + [32.]  * wzones + list(np.array(contingency)+16) + [1024.])

#%%
from Simulation import Reliability
from Network import Transmission

@njit()
def F(S):
    
    Deficit = Reliability(S, flexible=np.zeros((intervals, ) , dtype=np.float64)) # Sj-EDE(t, j), MW
    Flexible = Deficit.sum(axis=0) * resolution / years / efficiency # MWh p.a.
    Hydro = Flexible + GBaseload.sum() * resolution / years # Hydropower & biomass: MWh p.a.
    PenHydro = np.maximum(0, Hydro - 20_000_000) # TWh p.a. to MWh p.a.

    Deficit = Reliability(S, flexible=np.ones((intervals, ), dtype=np.float64)*CPeak.sum()*1000) # Sj-EDE(t, j), GW to MW
    PenDeficit = np.maximum(0, Deficit.sum(axis=0) * resolution) # MWh

    TDC_abs = np.abs(Transmission(S)) if scenario>=21 else np.zeros((intervals, len(DCloss)), dtype=np.float64)  # TDC: TDC(t, k), MW

    CDC = np.zeros(len(DCloss), dtype=np.float64)
    for j in prange(len(DCloss)):
        for i in range(intervals):
            CDC[j] = np.maximum(TDC_abs[i, j], CDC[j])
    CDC = CDC * 0.001 # CDC(k), MW to GW
    PenDC = max(0, CDC[6] - CDC6max) * 0.001 # GW to MW

    _c = 0 if scenario <= 17 else -1
    cost = (factor * np.array([S.CPV.sum(), S.CWind.sum(), S.CPHP.sum(), S.CPHS] + list(CDC) +
                              [S.CPV.sum(), S.CWind.sum(), Hydro * 0.000_001, _c, _c])
            )

    loss = TDC_abs.sum(axis=0) * DCloss
    loss = loss.sum(axis=0) * 0.000_000_001 * resolution / years # PWh p.a.
    energyloss = np.abs(energy - loss)
    LCOE = cost.sum() / energyloss
    LCOG = 1000 * cost[np.array([0, 1, 13])].sum() / (
        0.000_001*(resolution/years*(S.GPV.sum() + S.GWind.sum()) + Hydro))
    LCOBS = cost[np.array([2,3,14])].sum()/energyloss
    LCOBT = cost[np.array([4,5,6,7,8,9,10,11,12,15])].sum()/energyloss
    LCOBL = LCOE - LCOG - LCOBS - LCOBT
    
    return LCOE, (PenHydro+PenDeficit+PenDC), LCOG, LCOBS, LCOBT, LCOBL

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int64),
    ('nodes', int64),
    ('resolution',float64),
    ('CPV', float64[:]), # 1D array of floats
    ('CWind', float64[:]), # 1D array of floats
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GWind', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:,]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('Windl_int', int64[:]),
    ('GBaseload', float64[:, :]),  # 2D array of floats
    ('CPeak', float64[:]),  # 1D array of floats
    ('CHydro', float64[:]),  # 1D array of floats
    ('flexible', float64[:]),
    ('Discharge', float64[:]),
    ('Charge', float64[:]),
    ('Storage', float64[:]),
    ('Deficit', float64[:]),
    ('Spillage', float64[:]),
    ('Netload' ,float64[:]),
    ('Penalties', float64),
    ('LCOE', float64),
    ('LCOG', float64),
    ('LCOBS', float64),
    ('LCOBT', float64),
    ('LCOBL', float64),
    ('evaluated', boolean),
    ('MPV', float64[:, :]),
    ('MWind', float64[:, :]),
    ('MBaseload', float64[:, :]),
    ('MPeak', float64[:, :]),
    ('MDischarge', float64[:, :]),
    ('MCharge', float64[:, :]),
    ('MStorage', float64[:, :]),
    ('MDeficit', float64[:, :]),
    ('MSpillage', float64[:, :]),
    ('MHydro', float64[:, :]),
    ('MBio', float64[:, :]),
    ('TDC', float64[:, :]),
    ('CDC', float64[:]),
    ('FQ', float64[:]),
    ('NQ', float64[:]),
    ('NS', float64[:]),
    ('NV', float64[:]),
    ('AS', float64[:]),
    ('SW', float64[:]),
    ('TV', float64[:]),
    ('Topology', float64[:, :]),
]

@jitclass(solution_spec)
class Solution:
    #A candidate solution of decision variables CPV(i), CWind(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        # input vector should have shape (sidx+1, n) i.e. vertical input vectors
        assert len(x) == len(lb)
        
        self.x = x
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
       
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.CWind = x[pidx: widx]  # CWind(i), GW

        self.GPV = TSPV * np.ones((intervals, len(self.CPV))) * self.CPV * 1000.  # GPV(i, t), GW to MW
        self.GWind = TSWind * np.ones((intervals, len(self.CWind))) * self.CWind * 1000.  # GWind(i, t), GW to MW

        self.CPHP = x[widx: sidx]  # CPHP(j), GW
        self.CPHS = x[sidx]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.Windl_int = Nodel_int, PVl_int, Windl_int
        
        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        
        self.evaluated=False
        
    def _evaluate(self):
        self.LCOE, self.Penalties, self.LCOG, self.LCOBS, self.LCOBT, self.LCOBL = F(self)
        self.evaluated=True

    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    solution = Solution(x)#/1.25) 
    solution._evaluate()
    print(solution.LCOE, solution.Penalties)
    print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)

    
    def test(printout=True):
        x = np.random.rand(len(lb))*(ub-lb)+lb
        solution = Solution(x)#/1.25) 
        solution._evaluate()
        if printout:
            print(solution.LCOE, solution.Penalties)
            print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
    # test()
        
