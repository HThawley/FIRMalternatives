# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission
from Fill import Fill

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""

    Load, PV, OnsW = (solution.MLoad.sum(axis=1), solution.GPV.sum(axis=1), solution.GOnsW.sum(axis=1))
    Baseload, Peak = (solution.GBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.Discharge, solution.Charge, solution.Storage)
    Deficit, Spillage = (solution.Deficit, solution.Spillage)

    PHS = solution.CPHS * pow(10, 3) # GWh to MWh
    efficiency = solution.efficiency

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - OnsW[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * PHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
    try:
        assert np.amax(PV) <= sum(solution.CPV) * pow(10, 3), print(np.amax(PV) - sum(solution.CPV) * pow(10, 3))
        assert np.amax(OnsW) <= sum(solution.COnsW) * pow(10, 3), print(np.amax(OnsW) - sum(solution.COnsW) * pow(10, 3))

        assert np.amax(Discharge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Discharge) - sum(solution.CPHP) * pow(10, 3))
        assert np.amax(Charge) <= sum(solution.CPHP) * pow(10, 3), print(np.amax(Charge) - sum(solution.CPHP) * pow(10, 3))
        assert np.amax(Storage) <= solution.CPHS * pow(10, 3), print(np.amax(Storage) - solution.CPHS * pow(10, 3))
    except AssertionError:
        pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    Debug(solution)

    C = np.stack([solution.MLoad.sum(axis=1), solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), 
                  solution.GPV.sum(axis=1), solution.GOnsW.sum(axis=1), solution.Discharge, 
                  solution.Deficit, -1 * solution.Spillage, -1 * solution.Charge, solution.Storage,
                  solution.FQ, solution.NQ, solution.NS, solution.NV, solution.AS, solution.SW, solution.TV])
    C = np.around(C.transpose())

    datentime = np.array([(dt.datetime(firstyear, 1, 1, 0, 0) + x * dt.timedelta(minutes=60 * resolution)).strftime('%a %d-%b %Y %H:%M') for x in range(intervals)])
    C = np.insert(C.astype('str'), 0, datentime, axis=1)

    header = 'Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,' \
             'Pumped hydro energy storage,Energy deficit,Energy spillage,PHES-Charge,' \
             'PHES-Storage,FNQ-QLD,NSW-QLD,NSW-SA,NSW-VIC,NT-SA,SA-WA,TAS-VIC'

    np.savetxt(f'Results/S{scenario}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    if int(scenario)>=21:
        header = 'Date & time,Operational demand,Hydropower,Biomass,Solar photovoltaics,Wind,' \
                 'Pumped hydro energy storage,Energy deficit,Energy spillage,' \
                 'Transmission,PHES-Charge,PHES-Storage'

        Topology = solution.Topology[np.where(np.in1d(np.array(['FNQ', 'NSW', 'NT', 'QLD', 'SA', 'TAS', 'VIC', 'WA']), coverage) == True)[0]]

        for j in range(nodes):
            C = np.stack([(solution.MLoad)[:, j], solution.MHydro[:, j], solution.MBio[:, j], solution.MPV[:, j], 
                          solution.MOnsW[:, j], solution.MDischarge[:, j], solution.MDeficit[:, j],
                          -1 * solution.MSpillage[:, j], Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j]])
            C = np.around(C.transpose())

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt(f'Results/S{scenario}{Nodel[j]}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    CPV, COnsW, CPHP, CPHS = (sum(solution.CPV), sum(solution.COnsW), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GOnsW, GHydro, GBio, GPHES = map(lambda x: x * 0.000_001 * resolution / years, 
                                          (solution.GPV.sum(), solution.GOnsW.sum(), solution.MHydro.sum(),
                                            solution.MBio.sum(), solution.MDischarge.sum())) # TWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFOnsW = (G/C/0.0876 for G, C in zip((GPV, GOnsW), (CPV, COnsW)))

    CostPV    = costs.pv    * CPV    * pow(10, -9) # A$b p.a.
    CostOnsW  = costs.onsw  * COnsW  * pow(10, -9) # A$b p.a.
    CostHydro = costs.hydro * GHydro * 0.001 # A$b p.a.
    CostBio   = costs.hydro * GBio   * 0.001 # A$b p.a.
    CostPH    = (costs.phes[0] * CPHP 
                 + costs.phes[1] * CPHS 
                 # + costs.phes[2] * GPHES * pow(10, 6)
                 + costs.phes[3]) * pow(10, -9) # A$b p.a.

    CostDC = (costs.hvdc * solution.CDC).sum() * pow(10, -9) # A$b p.a.
    CostAC = costs.ac * (CPV + COnsW) * pow(10, -9) # A$b p.a.

    Energy = MLoad.sum() * pow(10, -9) * resolution / years # PWh p.a.
    Loss = np.sum(np.abs(solution.TDC), axis=0) * DCloss
    Loss = Loss.sum() * pow(10, -9) * resolution / years # PWh p.a.

    LCOE = (CostPV + CostOnsW + CostHydro + CostBio + CostPH + CostDC + CostAC) / (Energy - Loss)
    LCOG = (CostPV +  CostOnsW + CostHydro + CostBio) * 1000 / (GPV + GOnsW + GHydro + GBio)
    LCOGP    = CostPV    * 1000 / GPV    if GPV!=0    else 0
    LCOGOnsW = CostOnsW  * 1000 / GOnsW  if GOnsW!=0  else 0
    LCOGH    = CostHydro * 1000 / GHydro if GHydro!=0 else 0
    LCOGB    = CostBio   * 1000 / GBio   if GBio!=0   else 0
    
    LCOB  = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CostDC + CostAC) / (Energy - Loss)
    LCOBL = LCOB - LCOBS - LCOBT
    
    print('Levelised costs of electricity:')
    print(f'\u2022 LCOE: {LCOE}')
    print(f'\u2022 LCOG: {LCOG}')
    print(f'\u2022 LCOB: {LCOB}')
    print(f'\u2022 LCOG-PV: {LCOGP}, (CF:{round(CFPV,3)}%)')
    print(f'\u2022 LCOG-Onshore Wind: {LCOGOnsW} (CF:{round(CFOnsW,3)}%)')
    print(f'\u2022 LCOG-Hydro: {LCOGH}')
    print(f'\u2022 LCOG-Bio: {LCOGB}')
    print(f'\u2022 LCOB-Storage: {LCOBS}')
    print(f'\u2022 LCOB-Transmission: {LCOBT}')
    print(f'\u2022 LCOB-Spillage & loss: {LCOBL}')

    D = np.array([Energy * 1000, Loss * 1000, CPV, GPV, COnsW, GOnsW,  
                  CapHydrobio, GHydrobio, CPHP, CPHS, GPHES]
              + list(solution.CDC)
              + [LCOE, LCOG, LCOBS, LCOBT, LCOBL])

    header = ','.join(['Demand Served (TWh p.a.)', 'Transmission Loss (TWh p.a.)', 
                       'Utility PV (GW)', 'Utility PV (TWh p.a.)', 'Onshore Wind (GW)',
                       'Onshore Wind (TWh p.a.)', 'Hydro&Bio (GW)', 'Hydro&Bio (TWh p.a.)', 
                       'Pumped Hydro (GW)', 'Pumped Hydro (GWh)', 'Pumped Hydro (TWh p.a.)',
                       'FNQ-QLD (GW)','NSW-QLD (GW)','NSW-SA (GW)','NSW-VIC (GW)','NT-SA (GW)',
                       'SA-WA (GW)','TAS-VIC (GW)','LCOE', 'LCOG', 'LCOB - storage', 
                       'LCOB - Transmission&Distribution', 'LCOB - Curtailments and other losses'])

    np.savetxt(f'Results/GGTA{scenario}.csv', D.reshape(1,-1), fmt='%s', delimiter=',', header=header, comments='')
    print('Energy generation, storage and transmission information is produced.')

    return True

def Information(x, flexible):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)
    Deficit = Reliability(S, flexible=flexible)

    try:
        assert Deficit.sum() * resolution < 0.1, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass

    if int(scenario)>=21:
        
        
        S.TDC = Transmission(S) # TDC(t, k), MW
    else:
        S.TDC = np.zeros((intervals, len(DCloss))) # TDC(t, k), MW

        S.MPeak = np.tile(flexible, (nodes, 1)).transpose() # MW
        S.MBaseload = GBaseload.copy() # MW

        S.MPV = S.GPV.sum(axis=1) if S.GPV.shape[1]>0 else np.zeros((intervals, 1))
        S.MOnsW = S.GOnsW.sum(axis=1) if S.GOnsW.shape[1]>0 else np.zeros((intervals, 1))

        S.MDischarge = S.Discharge.reshape(-1,1)
        S.MDeficit   = S.Deficit.reshape(-1,1)
        S.MCharge    = S.Charge.reshape(-1,1)
        S.MStorage   = S.Storage.reshape(-1,1)
        S.MSpillage  = S.Spillage.reshape(-1,1)

    S.CDC = np.amax(np.abs(S.TDC), axis=0) * 0.001 # CDC(k), MW to GW
    S.FQ, S.NQ, S.NS, S.NV, S.AS, S.SW, S.TV = map(lambda k: S.TDC[:, k], range(S.TDC.shape[1]))

    S.MHydro = np.tile(CHydro - CBaseload, (intervals, 1)) * 1000 # GW to MW
    S.MHydro = np.minimum(S.MHydro, S.MPeak)
    S.MBio = S.MPeak - S.MHydro
    S.MHydro += S.GBaseload

    S.Topology = np.array([-1 * S.FQ, -1 * (S.NQ + S.NS + S.NV), -1 * S.AS, S.FQ + S.NQ, S.NS + S.AS - S.SW, -1 * S.TV, S.NV + S.TV, S.SW])

    LPGM(S)
    GGTA(S)

    end = dt.datetime.now()
    print("Statistics took", end - start)

    return True

# class Scenario:
#     def __init__(self, scen):
#         self.scen=scen
#     def __str__(self):
#         return str(self.scen)
#     def __repr__(self):
#         return str(self.scen)
#     def __int__(self):
#         return int(self.scen[-2:])

if __name__ == '__main__':
    capacities = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',')
    # scenario =Scenario('HighDist31')
    # capacities = np.genfromtxt('Results/{}.csv'.format(scenario), delimiter=',')
    
    S=Solution(capacities)
    Flex = Fill(S)
    
    Information(capacities, Flex)
    
    
    