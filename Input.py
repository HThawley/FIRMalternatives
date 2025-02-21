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

parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')

parser.add_argument('-cb', default=2, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-resume', default=0, type=int, required=False, help='Boolean - whether to restart')

args = parser.parse_args()
scenario = args.s

from Costs import cost_factors

Nodel = np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA'])
PVl =   np.array(['NSW']*7 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*3 + ['SA']*6 + ['TAS']*0 + ['VIC']*1 + ['WA']*1 + ['NT']*1)
OnsWl = np.array(['NSW']*8 + ['FNQ']*1 + ['QLD']*2 + ['FNQ']*2 + ['SA']*8 + ['TAS']*4 + ['VIC']*4 + ['WA']*3 + ['NT']*1)

n_node = dict((name, i) for i, name in enumerate(Nodel))
Nodel_int, PVl_int, OnsWl_int = (np.array([n_node[node] for node in x]).astype(np.int64) for x in (Nodel, PVl, OnsWl))

resolution = 0.5
firstyear, finalyear, timestep = (2020, 2029, 1)

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) * 0.001 # GW

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) # CFs (unitless)
TSOnsW = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) # CFs (unitless)

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * pow(10, -3) for x in range(assets.shape[1])] # GW
CBaseload = np.array([0, 0, 0, 0, 0, 1.0, 0, 0]) # GW
CPeak = CHydro + CBio - CBaseload # GW

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400]) #km
DCloss = DClengths * 0.03 * pow(10, -3) # unitless
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)
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

undersea_mask = undersea_mask[network_mask]
    
intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSOnsW.shape[1])
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + nodes)

energy = MLoad.sum() * pow(10, -6) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

lb = np.array([0.]  * pzones + [0.]   * wzones + contingency   + [0.])
ub = np.array([32.] * pzones + [32.]  * wzones + nodes*[32.] + [1024.])

#%%

costs = cost_factors(DClengths, undersea_mask)
# pre-allocating memory will save time on future evaluation with jit
flex_min = np.zeros(intervals, dtype=np.float64)
flex_max = np.ones(intervals,  dtype=np.float64)*CPeak.sum()
GBase = CBaseload.sum()*intervals*resolution/years
TDC_empty = np.zeros((intervals, len(DCloss)), dtype=np.float64)
# eff_fac = (0.5*(1+efficiency))
flex_fac = resolution/years/efficiency

from Simulation import Reliability
from Network import Transmission

# Specify the types for jitclass
solution_spec = [
    # general
    ('intervals', int64),
    ('resolution', float64),
    ('efficiency', float64),
    
    #scenario set-up
    ('scenario', int64),
    ('nodes', int64),
    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('OnsWl_int', int64[:]),
    
    # capacities
    ('x', float64[:]), 
    ('CPV', float64[:]), 
    ('COnsW', float64[:]),
    ('CPHP', float64[:]),
    ('CPHS', float64),
    ('CBaseload', float64[:]),
    ('CPeak', float64[:]), 
    ('CHydro', float64[:]),
    ('CDC', float64[:]),
    ('GCPHP', float64),# =CPHP.sum() avoid redundant computation

    # grid-level behaviour
    ('GFlexible', float64[:]),
    ('GDischarge', float64[:]),
    ('GCharge', float64[:]),
    ('GStorage', float64[:]),
    ('GDeficit', float64[:]),
    ('GSpillage', float64[:]),
    ('GNetload', float64[:]),
    
    # state-level behaviour
    ('MLoad', float64[:, :]), 
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
    ('MImport', float64[:, :]),
    
    # transmission behaviour
    ('TDC', float64[:, :]),
    ('FQ', float64[:]),
    ('NQ', float64[:]),
    ('NS', float64[:]),
    ('NV', float64[:]),
    ('AS', float64[:]),
    ('SW', float64[:]),
    ('TV', float64[:]),
    
    # objectives
    ('Penalties', float64),
    ('LCOE', float64),
    ('LCOG', float64),
    ('LCOBS', float64),
    ('LCOBT', float64),
    ('LCOBL', float64),
    ('evaluated', boolean),
]

@jitclass(solution_spec)
class Solution:
    def __init__(self, x):
        assert len(x) == len(lb)
        self.x, self.scenario = x, scenario

        self.Nodel_int, self.PVl_int, self.OnsWl_int = Nodel_int, PVl_int, OnsWl_int
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution = resolution
        self.efficiency = efficiency

        self.CPV   = x[: pidx]
        self.COnsW = x[pidx: widx]
        self.CPHP  = x[widx: sidx]
        self.CPHS  = x[sidx]
        self.CBaseload, self.CPeak, self.CHydro = CBaseload, CPeak, CHydro
        self.GCPHP = self.CPHP.sum()

        self.MPV, self.MOnsW = np.zeros((intervals, nodes)), np.zeros((intervals, nodes))
        for i, n in enumerate(self.Nodel_int):
            self.MPV[:, i] += (TSPV[:, PVl_int==n] * self.CPV[PVl_int==n]).sum(axis=1)
            self.MOnsW[:, i] += (TSOnsW[:, OnsWl_int==n] * self.COnsW[OnsWl_int==n]).sum(axis=1)
        self.MLoad = MLoad

        self.evaluated=False
        
    def _evaluate(self, costs):
        Hydro = GBase + Reliability(self, flexible=flex_min).sum() * flex_fac 
        self.Penalties = max(0, Hydro - 20_000_000) 
        self.Penalties += max(0, Reliability(self, flexible=flex_max).sum() * self.resolution) 

        TDC = np.abs(Transmission(self)) if self.scenario>=21 else TDC_empty

        CDC = np.zeros(len(DCloss), dtype=np.float64)
        for j in prange(len(DCloss)):
            for i in range(intervals):
                CDC[j] = np.maximum(TDC[i, j], CDC[j])
        # Penatlies += max(0, CDC[6] - CDC6max) 

        cost = np.array([
            self.CPV.sum() * costs.pv, 
            self.COnsW.sum() * costs.onsw, 
            (self.CPV.sum() + self.COnsW.sum())*costs.ac,
            self.CPHP.sum() * costs.phes[0],
            self.CPHS * costs.phes[1],
            0,# S.GDischarge.sum() * 1000 * costs.phes[2] * resolution / years + 
            costs.phes[3],] +
            list(CDC * costs.hvdc) +
            [Hydro * costs.hydro * 1000,
            ]) / 1_000_000_000 # $billiions p.a.
                
        energyloss = np.abs(energy - (TDC.sum(axis=0) * DCloss * 0.000_001).sum() * resolution / years)
        self.LCOE = cost.sum() / energyloss
        self.LCOG = 1000 * (cost[0]+cost[1]+cost[14]) / (
            0.001*(resolution/years*(self.MPV.sum() + self.MOnsW.sum()) + Hydro))
        self.LCOBS = (cost[3]+cost[4]+cost[5]+cost[6])/energyloss
        self.LCOBT = (cost[2]+cost[7]+cost[8]+cost[9]+cost[10]+cost[11]+cost[12]+cost[13])/energyloss
        self.LCOBL = self.LCOE - self.LCOG - self.LCOBS - self.LCOBT


if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    # x = np.random.rand(len(lb))*(ub-lb)+lb
    solution = Solution(x)
    solution._evaluate(costs)
    print(solution.LCOE, solution.Penalties)
    print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)

    @njit
    def test(costs, disp=False):
        x = np.random.rand(len(lb))*(ub-lb)+lb
        solution = Solution(x)
        solution._evaluate(costs)
        if disp:
            print(solution.LCOE, solution.Penalties)
            print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
            
    
    test(costs)
        
