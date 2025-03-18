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
from Fill import Fill


parser = ArgumentParser()
# scenario
parser.add_argument('-s', default=31, type=int, required=False, help='11, 12, 13, ...')

# mga
parser.add_argument('-cb', default=2, type=int, required=False, help='Callback: 0-None, 1-generation elites, 2-everything')
parser.add_argument('-ver', default=1, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-resume', default=0, type=int, required=False, help='Boolean - whether to restart')
parser.add_argument('-mp', default='jit', type=str, required=False, help='Multiprocessing method: pool or jit')
parser.add_argument('-d', default=-1, type=int, required=False, help='Number of splitting dimensions')

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

MLoad = np.genfromtxt('Data/electricity.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(Nodel))) /1000

TSPV = np.genfromtxt('Data/pv.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(PVl))) 
TSOnsW = np.genfromtxt('Data/wind.csv', delimiter=',', skip_header=1, usecols=range(4, 4+len(OnsWl))) 

assets = np.genfromtxt('Data/hydrobio.csv', dtype=None, delimiter=',', encoding=None)[1:, 1:].astype(float)
CHydro, CBio = [assets[:, x] * 0.001 for x in range(assets.shape[1])] 
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


elif scenario >= 21:
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


if 31 <= scenario <= 40:
    TSPV = np.stack([TSPV[:, PVl==node].mean(axis=1) for node in np.unique(PVl)]).T
    TSOnsW = np.stack([TSOnsW[:, OnsWl==node].mean(axis=1) for node in np.unique(OnsWl)]).T
    
    Nodel_int, PVl_int, OnsWl_int = [np.unique(x) for x in (Nodel_int, PVl_int, OnsWl_int)]
    Nodel, PVl, OnsWl = [np.unique(x)  for x in (Nodel, PVl, OnsWl)]
    
    pv_lb, pv_ub = np.zeros(len(Nodel), np.float64), 32*np.ones(len(Nodel), np.float64)

if scenario >= 41:
    pcfs = np.array([TSPV.mean(axis=0)[np.in1d(PVl, node)].max() for node in np.unique(PVl)])
    cfidxs = (np.array([TSPV.mean(axis=0)[np.in1d(PVl, node)].argmax() for node in np.unique(PVl)]) +
           np.array([0]+[np.in1d(PVl, node).sum() for node in np.unique(PVl)][:-1]).cumsum())
    TSPV = TSPV[:, cfidxs]#.mean(axis=1).reshape(-1, 1)
   
    wcfs = np.array([TSOnsW.mean(axis=0)[np.in1d(OnsWl, node)].max() for node in np.unique(OnsWl)])
    cfidxs = (np.array([TSOnsW.mean(axis=0)[np.in1d(OnsWl, node)].argmax() for node in np.unique(OnsWl)]) +
           np.array([0]+[np.in1d(OnsWl, node).sum() for node in np.unique(OnsWl)][:-1]).cumsum())
    TSOnsW = TSOnsW[:, cfidxs]#.mean(axis=1).reshape(-1, 1)
   
    pcfs = pcfs / pcfs.sum()
    wcfs = wcfs / wcfs.sum()
   
    pv_lb, pv_ub = np.array([0]), np.array([32.])
    
    
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)[network_mask]

mload = MLoad.sum(axis=0) 
pmask = np.isin(Nodel_int, PVl_int)
wmask  = np.isin(Nodel_int, OnsWl_int)
pload_factor =  mload[pmask] / mload[pmask].sum()
wload_factor =  mload[wmask] / mload[wmask].sum()

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
energy = MLoad.sum() * resolution / years # GWh p.a.
MBaseload = np.tile(CBaseload, (intervals, 1)) 

pzones, wzones = (len(PVl), len(OnsWl))
pnodes, wnodes, phnodes = nodes, nodes, nodes
maxim = 32.
if scenario >=41:
    pnodes, wnodes, phnodes = pmask.sum(), wmask.sum(), 1
    pzones, wzones = 1, 1
    scale=nodes
    maxim = 128.
pidx, widx, sidx = (pzones, pzones + wzones, pzones + wzones + phnodes)



lb = np.array([0.]*pzones +        [0.]*wzones +        [0.]*phnodes  +       [0.])
ub = np.array([maxim]*pzones + [maxim]*wzones + [maxim]*phnodes + [1024.])



#%%

costs = cost_factors(DClengths, undersea_mask)

# Specify the types for jitclass
solution_spec = [
    ('x', float64[:]), 
    ('scenario', int64), 
    
    ('intervals', int64),
    ('nodes', int64),
    ('nhvdc', int64),
    ('resolution', float64),
    ('years', float64),
    ('efficiency', float64),

    ('Nodel_int', int64[:]), 
    ('PVl_int', int64[:]),
    ('OnsWl_int', int64[:]),

    ('CPV', float64[:]),
    ('COnsW', float64[:]),
    ('CPHP', float64[:,]),
    ('CPHS', float64),
    ('CPeak', float64[:]),

    ('GCPHP', float64),
    ('GCPeak', float64),
    
    ('GFlexible', float64[:]),
    ('GDischarge', float64[:]),
    ('GCharge', float64[:]),
    ('GStorage', float64[:]),
    ('GDeficit', float64[:]),
    ('GSpillage', float64[:]),
    ('GNetload' ,float64[:]),

    ('MLoad', float64[:, :]),
    ('MBaseload', float64[:, :]),
    ('MOnsW', float64[:, :]),
    ('MPV', float64[:, :]),
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
    # ('FQ', float64[:]),
    # ('NQ', float64[:]),
    # ('NS', float64[:]),
    # ('NV', float64[:]),
    # ('AS', float64[:]),
    # ('SW', float64[:]),
    # ('TV', float64[:]),
    ('Topology', float64[:, :]),
    
    ('Penalties', float64),
    ('LCOE', float64),
    ('LCOG', float64),
    ('LCOBS', float64),
    ('LCOBT', float64),
    ('LCOBL', float64),

]

@jitclass(solution_spec)
class Solution:
    def __init__(self, x):
        assert len(x) == len(lb)
        
        self.x = x
        self.scenario = scenario
        self.intervals, self.nodes = intervals, nodes
        self.nhvdc = len(DCloss)
        self.resolution, self.years = resolution, years
        self.efficiency = efficiency

        self.Nodel_int, self.PVl_int, self.OnsWl_int = Nodel_int, PVl_int, OnsWl_int
        
        self.MLoad = MLoad
        self.MBaseload = MBaseload

        self.CPV = x[: pidx] 
        self.COnsW = x[pidx: widx] 
        self.CPHP = x[widx: sidx]

        if self.scenario < 20:
            self.MPV = np.atleast_2d((TSPV * self.CPV).sum(axis=1)).T
            self.MOnsW = np.atleast_2d((TSOnsW * self.COnsW).sum(axis=1)).T

        if self.scenario >= 21 and self.scenario < 30:
            self.MPV, self.MOnsW = np.zeros((self.intervals, self.nodes)), np.zeros((self.intervals, self.nodes))
            for i, node in enumerate(self.Nodel_int):
                self.MPV[:, i] = (TSPV[:, self.PVl_int==node] * self.CPV[self.PVl_int==node]).sum(axis=1)
                self.MOnsW[:, i] = (TSOnsW[:, self.OnsWl_int==node] * self.COnsW[self.OnsWl_int==node]).sum(axis=1)
            
        if self.scenario >= 31 and self.scenario < 40:
            self.MPV, self.MOnsW = np.zeros((self.intervals, self.nodes)), np.zeros((self.intervals, self.nodes))
            self.MPV[:,   np.isin(self.Nodel_int, self.PVl_int)]   = TSPV   * self.CPV   
            self.MOnsW[:, np.isin(self.Nodel_int, self.OnsWl_int)] = TSOnsW * self.COnsW 
            
        if self.scenario >= 41:
            
            self.MPV, self.MOnsW = np.zeros((self.intervals, self.nodes)), np.zeros((self.intervals, self.nodes))
            # distribute according to annual load
            self.MPV[:, pmask] = TSPV * self.CPV * pload_factor
            self.MOnsW[:, wmask] = TSOnsW * self.COnsW * wload_factor
            
            #distribute according to capacity factor 
            # self.MPV[:, pmask] = TSPV * self.CPV * pcfs
            # self.MOnsW[:, wmask] = TSOnsW * self.COnsW * wcfs
            
            self.CPHP = x[widx: sidx]/mload.sum()*mload
        
        self.GCPHP = self.CPHP.sum()
        self.CPHS = x[sidx]

        self.CPeak = CPeak
        self.GCPeak = self.CPeak.sum()
    
    def _evaluate(self, costs):
        Hydro = (self.MBaseload.sum() + Fill(self).sum())*self.resolution/self.years
        self.Penalties = max(0, Hydro - 20_000_000) # 20 TWh p.a. 
        self.Penalties += max(0, self.GDeficit.sum() * self.resolution) 

        TDC = np.abs(Transmission(self)) if self.scenario>=21 else np.zeros((1, self.nhvdc))

        self.CDC = np.zeros(self.nhvdc, dtype=np.float64)
        for j in range(self.nhvdc):
            for i in range(len(TDC)):
                self.CDC[j] = np.maximum(TDC[i, j], self.CDC[j])

        cost = np.array([
            self.CPV.sum() * costs.pv, 
            self.COnsW.sum() * costs.onsw, 
            (self.CPV.sum() + self.COnsW.sum())*costs.ac,
            self.CPHP.sum() * costs.phes[0],
            self.CPHS * costs.phes[1],
            self.GDischarge.sum() * 1000 * costs.phes[2] * self.resolution / self.years,  
            costs.phes[3],] +
            list(self.CDC * costs.hvdc) +
            [Hydro * costs.hydro * 1000.,
            ]) # $ p.a.
                
        energyloss = 1000*np.abs(energy - (TDC.sum(axis=0) * DCloss).sum() * self.resolution / self.years) #MWh
        self.LCOE = cost.sum() / energyloss # $/MWh
        self.LCOG = (cost[0]+cost[1]+cost[14]) / (
            1000 * (self.resolution/self.years*(self.MPV.sum() + self.MOnsW.sum()) + Hydro))
        self.LCOBS = (cost[3]+cost[4]+cost[5]+cost[6]) / energyloss
        self.LCOBT = (cost[2]+cost[7]+cost[8]+cost[9]+cost[10]+cost[11]+cost[12]+cost[13]) / energyloss
        self.LCOBL = self.LCOE - self.LCOG - self.LCOBS - self.LCOBT
        
#%%
if __name__=='__main__':
    def test(printout=False):
        x = np.random.rand(len(lb))*(ub-lb)+lb
        solution = Solution(x)
        solution._evaluate(costs)
        if printout:
            print(solution.LCOE, solution.Penalties)
            print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
    
    try:
        if scenario == 41:
            x = np.array([42., 27.5, 21.22, 432.])
        elif scenario == 31:
            x = np.array([42./4]*4 + [27.5/5]*5 + [21.22/5]*5 + [432.])
            # x = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',', dtype=float)
        else: 
            raise KeyboardInterrupt
        solution = Solution(x)
        solution._evaluate(costs)
        print(solution.LCOE, solution.Penalties)
        print(solution.LCOE, solution.LCOG, solution.LCOBS, solution.LCOBT, solution.LCOBL)
    except (FileNotFoundError, KeyboardInterrupt):
        test(True)
        
