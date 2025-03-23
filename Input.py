import numpy as np
from numba import njit, float64, int64
from numba.experimental import jitclass
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', default=1000, type=int, required=False, help='maxiter=4000, 400')
parser.add_argument('-p', default=100, type=int, required=False, help='popsize=2, 10')
parser.add_argument('-ml', default=0.45, type=float, required=False, help='mutation lower=0.5')
parser.add_argument('-mu', default=0.55, type=float, required=False, help='mutation upper=0.5')
parser.add_argument('-r', default=0.3, type=float, required=False, help='recombination=0.3')

parser.add_argument('-s', default=21, type=int, required=False, help='11, 12, 13, ...')

parser.add_argument('-ver', default=0, type=int, required=False, help='Boolean - print progress to console')
parser.add_argument('-res', default=1, type=int, required=False, help='Boolean - whether to try resume')

args = parser.parse_args()
scenario = args.s

from Timekeeper import keeptime, timekeeper, timekeeper_names

from Costs import Raw_Costs
from Network import Transmission
from Fill import Fill

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

Hydro_resource = 16_000 # GWh p.a. # Annual resource limit
Hydro_cf = Hydro_resource / CHydro.sum()

# FQ, NQ, NS, NV, AS, SW, only TV constrained
DClengths = np.array([1500, 1000, 1000, 800, 1200, 2400, 400], np.int64) #km
DCloss = DClengths * 0.03 * pow(10, -3) # unitless
undersea_mask = np.array([0, 0, 0, 0, 0, 0, 1], dtype=bool)
CDC6max = 3 * 0.63 # GW

efficiency = 0.8
factor = np.genfromtxt('Data/factor.csv', delimiter=',', usecols=1)

if scenario<=17:
    node = Nodel[scenario % 10]
    network_mask = np.zeros(7, dtype=np.bool_)

    MLoad = MLoad[:, Nodel==node]
    TSPV = TSPV[:, PVl==node]
    TSOnsW = TSOnsW[:, OnsWl==node]
    CHydro, CBio, CBaseload = [x[Nodel==node] for x in (CHydro, CBio, CBaseload)]

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
    CHydro, CBio, CBaseload = [x[np.in1d(Nodel, coverage)] for x in (CHydro, CBio, CBaseload)]
    
    if 'FNQ' not in coverage:
        MLoad[:, np.where(coverage=='QLD')[0][0]] /= 0.9
        
    coverage_int = np.array([n_node[node] for node in coverage])
    Nodel_int, PVl_int, OnsWl_int = [x[np.isin(x, coverage_int)] for x in (Nodel_int, PVl_int, OnsWl_int)]
    Nodel, PVl, OnsWl = [x[np.isin(x, coverage)] for x in (Nodel, PVl, OnsWl)]

Hydro_resource = Hydro_cf * CHydro.sum()
Bio_resource = Hydro_cf*CBio.sum()

intervals, nodes = MLoad.shape
years = int(resolution * intervals / 8760)
pzones, wzones = (TSPV.shape[1], TSOnsW.shape[1])
pidx, widx, gidx, sidx = pzones, pzones + wzones, pzones + wzones + nodes, pzones + wzones + nodes*2

energy = MLoad.sum() * pow(10, -6) * resolution / years # PWh p.a.
contingency = list(0.25 * MLoad.max(axis=0) * pow(10, -3)) # MW to GW

lb = np.array([0.]  * pzones + [0.]   * wzones + [0.]  * nodes + contingency   + [0.])
ub = np.array([32.] * pzones + [32.]  * wzones + [32.] * nodes + nodes*[32.] + [1024.])

#%%

costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask).CostFactors()

# Specify the types for jitclass
solution_spec = [
    # general
    ('intervals',   int64),
    ('resolution',  float64),
    ('efficiency',  float64),
    ('years',       int64),
    ('Hydro_res',   float64),
    ('Bio_res',     float64),
    
    #scenario set-up
    ('scenario',    int64),
    ('nodes',       int64),
    ('Nodel_int',   int64[:]), 
    ('PVl_int',     int64[:]),
    ('OnsWl_int',   int64[:]),
    
    # capacities
    ('x',           float64[:]), 
    ('CPV',         float64[:]), 
    ('COnsW',       float64[:]),
    ('CGas',        float64[:]),
    ('CPHP',        float64[:]),
    ('CPHS',        float64),
    ('CBaseload',   float64[:]),
    ('CHydro',      float64[:]),
    ('CBio',        float64[:]),
    ('CDC',         float64[:]),
    ('GCPHP',       float64),# =CPHP.sum() avoid redundant computation

    # grid-level behaviour
    ('GFlexible',   float64[:]),
    ('GGas',        float64[:]),
    ('GHydro',      float64[:]),
    ('GBio',      float64[:]),
    ('GDischarge',  float64[:]),
    ('GCharge',     float64[:]),
    ('GStorage',    float64[:]),
    ('GDeficit',    float64[:]),
    ('GSpillage',   float64[:]),
    ('GNetload',    float64[:]),
    
    # state-level behaviour
    ('MLoad',       float64[:, :]), 
    ('MPV',         float64[:, :]),
    ('MOnsW',       float64[:, :]),
    ('MDischarge',  float64[:, :]),
    ('MCharge',     float64[:, :]),
    ('MStorage',    float64[:, :]),
    ('MDeficit',    float64[:, :]),
    ('MSpillage',   float64[:, :]),
    ('MHydro',      float64[:, :]),
    ('MBio',        float64[:, :]),
    ('MGas',        float64[:, :]),
    ('MImport',     float64[:, :]),
    
    # transmission behaviour
    ('TDC', float64[:, :]),
    ('FQ',  float64[:]),
    ('NQ',  float64[:]),
    ('NS',  float64[:]),
    ('NV',  float64[:]),
    ('AS',  float64[:]),
    ('SW',  float64[:]),
    ('TV',  float64[:]),
    
    # objectives
    ('Penalties',   float64),
    ('LCOE',        float64),
    ('LCOG',        float64),
    ('LCOBS',       float64),
    ('LCOBT',       float64),
    ('LCOBL',       float64),
    ('Capex',       float64),
    ('Opex',        float64),
    ('energyloss',  float64),
]

@jitclass(solution_spec)
class Solution:
    def __init__(self, x):
        assert len(x) == len(lb)
        self.x, self.scenario = x, scenario

        self.Nodel_int, self.PVl_int, self.OnsWl_int = Nodel_int, PVl_int, OnsWl_int
        
        self.intervals, self.nodes = intervals, nodes
        self.resolution, self.efficiency, self.years = resolution, efficiency, years
        #TODO: remove cbaseload from Hydro_res 
        self.Hydro_res, self.Bio_res = [res/resolution*years for res in (Hydro_resource, Bio_resource)]
        self.Hydro_res -= CBaseload.sum()*self.intervals
        
        self.CPV   = x[: pidx]
        self.COnsW = x[pidx: widx]
        self.CGas  = x[widx: gidx]
        self.CPHP  = x[gidx: sidx]
        self.CPHS  = x[sidx]
        self.CBaseload = CBaseload
        self.CHydro, self.CBio = CHydro - self.CBaseload, CBio
        self.GCPHP = self.CPHP.sum()

        self.MPV, self.MOnsW = np.zeros((intervals, nodes)), np.zeros((intervals, nodes))
        for i, n in enumerate(self.Nodel_int):
            self.MPV[:, i] += (TSPV[:, PVl_int==n] * self.CPV[PVl_int==n]).sum(axis=1)
            self.MOnsW[:, i] += (TSOnsW[:, OnsWl_int==n] * self.COnsW[OnsWl_int==n]).sum(axis=1)
        self.MLoad = MLoad

    def _evaluate(self, costs):
        self.Penalties += max(0, Fill(self).sum()*self.resolution)

        if self.scenario >= 21:
            TDC = np.abs(Transmission(self))
            self.CDC = np.zeros(len(network_mask), dtype=np.float64)
            for j in range(len(network_mask)):
                for i in range(self.intervals):
                    self.CDC[j] = np.maximum(TDC[i, j], self.CDC[j])
        else: 
            TDC = self.TDC = np.zeros((1, len(network_mask)), np.float64)
            self.CDC = np.zeros(len(network_mask), np.float64)
            
        # Penatlies += max(0, CDC[6] - CDC6max) 

        cost = np.array([
            # generation capex 
            self.CPV.sum()   * costs.pv[0], 
            self.COnsW.sum() * costs.onsw[0], 
            self.CGas.sum()  * costs.gas[0],
            (self.CHydro.sum()+self.CBio.sum()+self.CBaseload.sum()) * costs.hydro[0],
            
            # generation fom
            self.CPV.sum()   * costs.pv[1], 
            self.COnsW.sum() * costs.onsw[1], 
            self.CGas.sum()  * costs.gas[1],
            (self.CHydro.sum()+self.CBio.sum()+self.CBaseload.sum()) * costs.hydro[1],
            
            # generation vom
            # pv, onsw, offw are 0
            self.GGas.sum() * self.resolution / self.years * costs.gas[2],
            (self.GHydro.sum() + self.GBio.sum() + self.CBaseload.sum()*self.intervals
             ) * self.resolution / self.years * costs.hydro[2],
            
            # storage 
            self.CPHP.sum() * costs.phes[0],
            self.CPHS * costs.phes[1],
            self.CPHP.sum() * costs.phes[2],
            self.GDischarge.sum() * self.resolution / self.years * costs.phes[3], 
            costs.phes[4],
            ] +
            
            # transmission network
            list((self.CPV.sum() + self.COnsW.sum() + self.CGas.sum() + self.CHydro.sum() 
             + self.CBio.sum())*costs.ac) +
            list((self.CDC * costs.hvdc).sum(axis=1))
            ) / 1_000_000_000 # $billions p.a.
        
        self.energyloss = np.abs(energy - (TDC.sum(axis=0) * DCloss * 0.000_001).sum() * self.resolution / self.years)
        self.LCOE = cost.sum() / self.energyloss
        self.LCOG = 1_000_000 * (cost[:10].sum()) / (self.resolution / self.years * (
            self.MPV.sum() + self.MOnsW.sum() + (self.GFlexible+self.CBaseload.sum()).sum()))
        self.LCOBS = cost[10:15].sum()/self.energyloss
        self.LCOBT = cost[15:].sum()/self.energyloss
        self.LCOBL = self.LCOE - self.LCOG - self.LCOBS - self.LCOBT
        self.Capex = sum([cost[i] for i in [0,1,2,3,10,11,14,15,18]])/self.energyloss
        self.Opex = sum([cost[i] for i in [4,5,6,7,8,9,12,13,16,17,19,20]])/self.energyloss
  
if __name__=='__main__':
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
    # x = np.random.rand(len(lb))*(ub-lb)+lb
    solution = Solution(x)
    #%%
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
        

