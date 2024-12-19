# Modelling input and assumptions
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

import numpy as np
from numba import njit, float64, int64, prange, boolean
from numba.experimental import jitclass
from argparse import ArgumentParser

from Costs import cost_factors
from Simulation import Reliability
from Network import Transmission


parser = ArgumentParser()
# scenario
parser.add_argument('-s', default=31, type=int, required=False, help='11, 12, 13, ...')

# mga
parser.add_argument('-cb', default=2, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-resume', default=0, type=int, required=False, help='Boolean - whether to restart')
parser.add_argument('-mp', default='jit', type=str, required=False, help='Multiprocessing method: pool or jit')

# DE
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-m', default=0.5, type=float, required=False, help='mutation=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')

args = parser.parse_args()
scenario = args.s


Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
OnsWl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, OnsWl_int = (np.array([n_node[node] for node in x], dtype=np.int64) for x in (Nodel, PVl, OnsWl))
Nodel_int, PVl_int, OnsWl_int = (x.astype(np.int64) for x in (Nodel_int, PVl_int, OnsWl_int))

resolution = 0.5
firstyear, finalyear, timestep = (2020, 2029, 1)

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) # EOLoad(t, j), MW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # TSPV(t, i), MW
TSOnsW = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) # TSOnsW(t, i), MW

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * 0.001 for x in range(assets.shape[1])] # CHydro(j), MW to GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # 24/7, GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) 
DCloss = DClengths * 0.03 * pow(10, -3)

CDC6max = 3 * 0.63 # GW

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

if scenario<=17:
    node = Nodel[scenario % 10]
    network_mask = np.array([], dtype=bool)

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSOnsW = TSOnsW[:, OnsWl==node]
    CHydro, CBio, CBaseload, CPeak = [x[Nodel==node] for x in (CHydro, CBio, CBaseload, CPeak)]

    Nodel_int, PVl_int, OnsWl_int = [x[x==n_node[node]] for x in (Nodel_int, PVl_int, OnsWl_int)]
    Nodel, PVl, OnsWl = [x[x==node] for x in (Nodel, PVl, OnsWl)]


elif scenario>=21:
    coverage = [np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'QLD', 'SA', 'TAS', 'VIC', 'WA']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC']),
                np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])][scenario % 10 - 1] 
    
    # 'FNQ-QLD', 'NSW-QLD', 'NSW-SA', 'NSW-VIC', 'NT-SA', 'SA-WA', 'TAS-VIC'
    network_mask = [np.array([0,1,1,1,0,0,1], dtype=bool),
                    np.array([0,1,1,1,0,1,1], dtype=bool),
                    np.array([0,1,1,1,1,0,1], dtype=bool),
                    np.array([0,1,1,1,1,1,1], dtype=bool),
                    np.array([1,1,1,1,0,1,1], dtype=bool),
                    np.array([1,1,1,1,1,0,1], dtype=bool),
                    np.array([1,1,1,1,1,1,1], dtype=bool)][scenario % 10 - 1] 
    
    MLoad = MLoad[:, np.in1d(Nodel, coverage)]
    TSPV = TSPV[:, np.in1d(PVl, coverage)]
    TSOnsW = TSOnsW[:, np.in1d(OnsWl, coverage)]
    CHydro, CBio, CBaseload, CPeak = [x[np.in1d(Nodel, coverage)] for x in (CHydro, CBio, CBaseload, CPeak)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, OnsWl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, OnsWl_int)]
    Nodel, PVl, OnsWl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, OnsWl)]

pv_lb, pv_ub = np.zeros(len(PVl), np.float64), 32.*np.ones(len(PVl), np.float64)

if scenario >= 31:
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    
    TSPV = np.stack([TSPV[:, PVl==node].mean(axis=1) for node in coverage]).T
    TSOnsW = np.stack([TSOnsW[:, OnsWl==node].mean(axis=1) for node in coverage]).T
    # having full of zeros and setting lb,ub=0,0 makes code faster
    TSPV = np.nan_to_num(TSPV, False, 0)
    warnings.simplefilter('default', RuntimeWarning)
    
    Nodel_int, PVl_int, OnsWl_int = [np.unique(x) for x in (Nodel_int, PVl_int, OnsWl_int)]
    Nodel, PVl, OnsWl = [np.unique(x)  for x in (Nodel, PVl, OnsWl)]
    
    pv_lb, pv_ub = np.zeros(len(Nodel), np.float64), 32*np.ones(len(Nodel), np.float64)
    pv_ub[3]=0
    
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)[network_mask]

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (len(PVl), len(OnsWl))
if scenario >= 31:
    pzones+=1
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

GBaseload = np.tile(CBaseload, (intervals, 1)) * pow(10, 3) # GW to MW

lb = np.array(list(pv_lb) + [0.]   * wzones + contingency  + [0.])
ub = np.array(list(pv_ub) + [32.]  * wzones + list(np.array(contingency)+16) + [1024.])

#%%

costs = cost_factors(DClengths, undersea_mask)
# pre-allocating memory will save time on future evaluation with jit
flex_min = np.zeros(intervals, dtype=np.float64)
flex_max = np.ones(intervals,  dtype=np.float64)*CPeak.sum()*1000
GBase = GBaseload.sum()*resolution/years
TDC_empty = np.zeros((intervals, len(DCloss)), dtype=np.float64)
# eff_fac = (0.5*(1+efficiency))
flex_fac = resolution/years/efficiency

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]),  # x is 1d array
    ('MLoad', float64[:, :]),  # 2D array of floats
    ('intervals', int64),
    ('nodes', int64),
    ('resolution',float64),
    ('CPV', float64[:]), # 1D array of floats
    ('COnsW', float64[:]), # 1D array of floats
    ('GPV', float64[:, :]),  # 2D array of floats
    ('GOnsW', float64[:, :]),  # 2D array of floats
    ('CPHP', float64[:,]),
    ('CPHS', float64),
    ('efficiency', float64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('OnsWl_int', int64[:]),
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
    ('MOnsW', float64[:, :]),
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
    #A candidate solution of decision variables CPV(i), COnsW(i), CPHP(j), S-CPHS(j)
    
    def __init__(self, x):
        # input vector should have shape (sidx+1, n) i.e. vertical input vectors
        assert len(x) == len(lb)
        
        self.x = x
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
       
        self.MLoad = MLoad

        self.CPV = x[: pidx]  # CPV(i), GW
        self.COnsW = x[pidx: widx]  # COnsW(i), GW

        self.GPV = TSPV * np.ones((intervals, len(self.CPV))) * self.CPV * 1000.  # GPV(i, t), GW to MW
        self.GOnsW = TSOnsW * np.ones((intervals, len(self.COnsW))) * self.COnsW * 1000.  # GOnsW(i, t), GW to MW

        self.CPHP = x[widx: sidx]  # CPHP(j), GW
        self.CPHS = x[sidx]  # S-CPHS(j), GWh
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.OnsWl_int = Nodel_int, PVl_int, OnsWl_int
        
        self.GBaseload = GBaseload
        self.CPeak = CPeak
        self.CHydro = CHydro
        
    def _evaluate(self, costs):
        Hydro = GBase + Reliability(self, flexible=flex_min).sum() * flex_fac 
        self.Penalties = max(0, Hydro - 20_000_000) 
        self.Penalties += max(0, Reliability(self, flexible=flex_max).sum() * resolution) 

        TDC = np.abs(Transmission(self)) if scenario>=21 else TDC_empty

        CDC = np.zeros(len(DCloss), dtype=np.float64)
        for j in prange(len(DCloss)):
            for i in range(intervals):
                CDC[j] = np.maximum(TDC[i, j], CDC[j])
        CDC = CDC * 0.001 # CDC(k), MW to GW
        # Penatlies += max(0, CDC[6] - CDC6max) * 0.001 # GW to MW

        cost = np.array([
            self.CPV.sum() * costs.pv, 
            self.COnsW.sum() * costs.onsw, 
            (self.CPV.sum() + self.COnsW.sum())*costs.ac,
            self.CPHP.sum() * costs.phes[0],
            self.CPHS * costs.phes[1],
            0,# S.Discharge.sum() * costs.phes[2] * resolution / years + 
            costs.phes[3],] +
            list(CDC * costs.hvdc) +
            [Hydro * costs.hydro,
            ]) / 1_000_000_000 # $billiions p.a.
                
        energyloss = np.abs(energy - (TDC.sum(axis=0) * DCloss).sum() * 0.000_000_001 * resolution / years)
        self.LCOE = cost.sum() / energyloss
        self.LCOG = 1000 * (cost[0]+cost[1]+cost[14]) / (
            0.000_001*(resolution/years*(self.GPV.sum() + self.GOnsW.sum()) + Hydro))
        self.LCOBS = (cost[3]+cost[4]+cost[5]+cost[6])/energyloss
        self.LCOBT = (cost[2]+cost[7]+cost[8]+cost[9]+cost[10]+cost[11]+cost[12]+cost[13])/energyloss
        self.LCOBL = self.LCOE - self.LCOG - self.LCOBS - self.LCOBT
        
    # def __repr__(self):
    #     """S = Solution(list(np.ones(64))) >> print(S)"""
    #     return 'Solution({})'.format(self.x)

if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    solution = Solution(x)#/1.25) 
    solution._evaluate(costs)
    print(solution.LCOE, solution.Penalties)
    print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)

    
    def test(printout=False):
        x = np.random.rand(len(lb))*(ub-lb)+lb
        solution = Solution(x)#/1.25) 
        solution._evaluate(costs)
        if printout:
            print(solution.LCOE, solution.Penalties)
            print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
    # test()
        
