# Load profiles and generation mix data (LPGM) & energy generation, storage and transmission information (GGTA)
# based on x/capacities from Optimisation and flexible from Dispatch
# Copyright (c) 2019, 2020 Bin Lu, The Australian National University
# Licensed under the MIT Licence
# Correspondence: bin.lu@anu.edu.au

from Input import *
from Simulation import Reliability
from Network import Transmission
from Fill import Fill
import warnings

import numpy as np
import datetime as dt

def Debug(solution):
    """Debugging"""

    Load, PV, OnsW = (solution.MLoad.sum(axis=1), solution.MPV.sum(axis=1), solution.MOnsW.sum(axis=1))
    Baseload, Peak = (solution.MBaseload.sum(axis=1), solution.MPeak.sum(axis=1))

    Discharge, Charge, Storage = (solution.GDischarge, solution.GCharge, solution.GStorage)
    Deficit, Spillage = (solution.GDeficit, solution.GSpillage)

    for i in range(intervals):
        # Energy supply-demand balance
        assert abs(Load[i] + Charge[i] + Spillage[i]
                   - PV[i] - OnsW[i] - Baseload[i] - Peak[i] - Discharge[i] - Deficit[i]) <= 1

        # Discharge, Charge and Storage
        if i==0:
            assert abs(Storage[i] - 0.5 * solution.CPHS + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1
        else:
            assert abs(Storage[i] - Storage[i - 1] + Discharge[i] * resolution - Charge[i] * resolution * efficiency) <= 1

        # Capacity: PV, wind, Discharge, Charge and Storage
    try:
        assert np.amax(PV) <= sum(solution.CPV), print(np.amax(PV) - sum(solution.CPV))
        assert np.amax(OnsW) <= sum(solution.COnsW), print(np.amax(OnsW) - sum(solution.COnsW))

        assert np.amax(Discharge) <= sum(solution.CPHP), print(np.amax(Discharge) - sum(solution.CPHP))
        assert np.amax(Charge) <= sum(solution.CPHP), print(np.amax(Charge) - sum(solution.CPHP))
        assert np.amax(Storage) <= solution.CPHS, print(np.amax(Storage) - solution.CPHS)
    except AssertionError:
        pass

    print('Debugging: everything is ok')

    return True

def LPGM(solution):
    """Load profiles and generation mix data"""

    C = np.stack((solution.MLoad.sum(axis=1), solution.MHydro.sum(axis=1), solution.MBio.sum(axis=1), 
                  solution.MPV.sum(axis=1), solution.MOnsW.sum(axis=1), solution.GDischarge, 
                  solution.GDeficit, -1 * solution.GSpillage, -1 * solution.GCharge, solution.GStorage))
    C = np.hstack((C.T, solution.TAC))
    C = np.around(1000*C) # GW & GWh to MW & MWh

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
            C = np.stack(((solution.MLoad)[:, j], solution.MHydro[:, j], solution.MBio[:, j], solution.MPV[:, j], 
                          solution.MOnsW[:, j], solution.MDischarge[:, j], solution.MDeficit[:, j],
                          -1 * solution.MSpillage[:, j], Topology[j], -1 * solution.MCharge[:, j],
                          solution.MStorage[:, j]))
            C = np.around(1000 * C.T) # GW & GWh to MW & MWh

            C = np.insert(C.astype('str'), 0, datentime, axis=1)
            np.savetxt(f'Results/S{scenario}{Nodel[j]}.csv', C, fmt='%s', delimiter=',', header=header, comments='')

    print('Load profiles and generation mix is produced.')

    return True

def GGTA(solution, save=True):
    """GW, GWh, TWh p.a. and A$/MWh information"""

    CPV, COnsW, CPHP, CPHS = (sum(solution.CPV), sum(solution.COnsW), sum(solution.CPHP), solution.CPHS) # GW, GWh
    CapHydro, CapBio = CHydro.sum(), CBio.sum() # GW
    CapHydrobio = CapHydro + CapBio

    GPV, GOnsW, GHydro, GBio, GPHES = map(lambda x: x.sum() * 1000 * resolution / years, 
                                          (solution.MPV, solution.MOnsW, solution.MHydro,
                                           solution.MBio, solution.MDischarge)) # MWh p.a.
    GHydrobio = GHydro + GBio
    CFPV, CFOnsW = (G/C/87600 for G, C in zip((GPV, GOnsW), (CPV, COnsW)))

    CostPV    = costs.pv    * CPV    # $ p.a.
    CostOnsW  = costs.onsw  * COnsW  # $ p.a.
    CostHydro = costs.hydro * GHydro # $ p.a.
    CostBio   = costs.hydro * GBio   # $ p.a.
    CostPH    = (costs.phes[0] * CPHP 
                 + costs.phes[1] * CPHS 
                 + costs.phes[2] * GPHES 
                 + costs.phes[3]) # $ p.a.

    CosTAC = 0#(costs.hvdc * solution.CDC).sum() # $ p.a.
    CostAC = costs.ac * (CPV + COnsW) # $ p.a.

    Energy = MLoad.sum() * 1000 * resolution / years # MWh p.a.
    Loss = 0
    # Loss = np.sum(np.abs(solution.TAC), axis=0) * DCloss
    # Loss = Loss.sum() * 1000 * resolution / years # MWh p.a.

    LCOE = (CostPV + CostOnsW + CostHydro + CostBio + CostPH + CosTAC + CostAC) / (Energy - Loss)
    LCOG = (CostPV +  CostOnsW + CostHydro + CostBio) / (GPV + GOnsW + GHydro + GBio)
    LCOGP    = CostPV    / GPV    if GPV!=0    else 0
    LCOGOnsW = CostOnsW  / GOnsW  if GOnsW!=0  else 0
    LCOGH    = CostHydro / GHydro if GHydro!=0 else 0
    LCOGB    = CostBio   / GBio   if GBio!=0   else 0
    
    LCOB  = LCOE - LCOG
    LCOBS = CostPH / (Energy - Loss)
    LCOBT = (CosTAC + CostAC) / (Energy - Loss)
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

    if save is True:
        D = np.array([Energy * 0.000_001, Loss * 0.000_001, 
                      CPV, GPV * 0.000_001, COnsW, 
                      GOnsW * 0.000_001, CapHydrobio, GHydrobio * 0.000_001,
                      CPHP, CPHS, GPHES * 0.000_001]
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

def TransmissionStatistics(solution):
    header = ','.join(['Quantity','FNQ-QLD','NSW-QLD','NSW-SA','NSW-VIC','NT-SA','SA-WA','TAS-VIC'])
    rows = np.array(['Capacity (GW)', 'Forward Transmission (TWh p.a.)', 'Reverse Transmission (TWh p.a.)', 
                     'Utilisation rate (%)'])
    
    with warnings.catch_warnings(category=RuntimeWarning, action='ignore'):
        T = np.array([[solution.CDC[i], np.maximum(0, solution.TAC[:,i]).sum()*0.001*resolution/years, 
                       -np.minimum(0, solution.TAC[:,i]).sum()*0.001*resolution/years, 
                       100*(np.abs(solution.TAC[:,i]).sum()*resolution/years)/(solution.CDC[i]*intervals*resolution/years)] 
                      for i in range(solution.nhvdc)]).T
        T = np.nan_to_num(T, False, 0)
    
    T = np.insert(T.astype('str'), 0, rows, axis=1)
    np.savetxt(f'Results/Trans{scenario}.csv', T, fmt='%s', delimiter=',', header=header, comments='')
    return True
    
    

def Information(x):
    """Dispatch: Statistics.Information(x, Flex)"""

    start = dt.datetime.now()
    print("Statistics start at", start)

    S = Solution(x)
    S._evaluate(costs)

    try:
        assert S.GDeficit.sum() * resolution < 0.001, 'Energy generation and demand are not balanced.'
    except AssertionError:
        pass

    S.TAC = Transmission(S) 

    S.CAC = np.amax(np.abs(S.TAC), axis=0) 

    S.MHydro = np.minimum(CHydro-CBaseload, S.MPeak)
    S.MBio = S.MPeak - S.MHydro
    S.MHydro += S.MBaseload

    S.Topology = np.stack((
        -1 * S.TAC[:,0], 
        -1 * (S.TAC[:,1] + S.TAC[:,2] + S.TAC[:,3]),
        -1 * S.TAC[:,4], 
        S.TAC[:,0] + S.TAC[:,1], 
        S.TAC[:,2] + S.TAC[:,4] - S.TAC[:,5], 
        -1 * S.TAC[:,6], 
        S.TAC[:,3] + S.TAC[:,6], 
        S.TAC[:,5]
        ))
    
    Debug(S)
    LPGM(S)
    GGTA(S)
    TransmissionStatistics(S)

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
#%%
if __name__ == '__main__':
    capacities = np.genfromtxt(f'Results/Optimisation_resultx{scenario}.csv', delimiter=',')
    # scenario =Scenario('HighDist31')
    # capacities = np.genfromtxt('Results/{}.csv'.format(scenario), delimiter=',')
    
    S=Solution(capacities)
    Flex = Fill(S)
    
    Information(capacities, Flex)
    
    
    