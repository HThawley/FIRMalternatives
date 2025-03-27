import numpy as np 
import pandas as pd
from numba import njit
from csv import writer


from Input import * 
from Costs import Raw_Costs 
from Optimisation import Optimise
from Timekeeper import keeptime, PrintTimekeeper, timekeeper
from Fileprinter import Fileprinter


## Parameters to sweep 
raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)

pv_capex = raw_costs.pv[0]
pv_capex = 0.75*pv_capex, pv_capex, 1.25*pv_capex

wind_capex = raw_costs.onsw[0]
wind_capex = 0.75*wind_capex, wind_capex, 1.25*wind_capex

gas_fuel = raw_costs.gas[3]
gas_fuel = 0.75*gas_fuel, gas_fuel, 1.25*gas_fuel, gas_fuel*1e9

carbon_price = (0, 35, 70, 140)

costs = raw_costs.CostFactors()

@keeptime('Select population')
def select_population(costs):
    history = read_history()
    history = deduplicate_history(history, commit=True, precision=4)
    history = history.to_numpy()
    Lcoes = calculate_costs(history, costs)
    sort_array = np.argsort(Lcoes)
    noptimaln = min(len(Lcoes), args.p)
    history = history[sort_array[:noptimaln], 14:]
    return history

@keeptime('Reading His.y')
def read_history():
    history = pd.read_csv(f'Results/History{scenario}.csv', header=None)
    return history

@keeptime('Dedup-ing History')
def deduplicate_history(history, commit=False, precision=None, subset=None):
    if precision is None:
        history = history.drop_duplicates(subset=subset)
    else: 
        history = history[~history.round(precision).duplicated(subset=subset)]
    if commit is True:
        write_history(history)
    return history
        
@keeptime('Write dedup-ed his.y')
def write_history(history):
    history.to_csv(f'Results/History{scenario}.csv', header=False, index=False)

@njit
def normalise(arr, lb, ub):
    return (arr-lb)/(ub-lb)

@njit
def unnormalise(arr, lb, ub):
    return arr*(ub-lb) + lb

@njit
def calculate_distances(history, centroid):
    distances = ((history - centroid)**2).sum(axis=1)**(1/2)
    return distances

@keeptime('Cost calcs')
@njit
def calculate_costs(history, costs):
    Lcoes = np.stack((
        (history[:,7:14] * costs.hvdc.sum(axis=0)).sum(axis=1), # hvdc capex and fom 
        history[:, 14:      14+pidx].sum(axis=1) * (costs.pv[0]   + costs.pv[1]   + costs.ac.sum()), # pv capex and fom
        history[:, 14+pidx: 14+widx].sum(axis=1) * (costs.onsw[0] + costs.onsw[1] + costs.ac.sum()), # wind capex and fom
        history[:, 14+widx: 14+gidx].sum(axis=1) * (costs.gas[0]  + costs.gas[1]  + costs.ac.sum()), # gas capex and fom
        history[:, 14+gidx: 14+sidx].sum(axis=1) * (costs.phes[0] + costs.phes[2] ), # phes capex (power) and fom
        history[:, 14+sidx] * costs.phes[1], # phes capex (energy)
        
        history[:, 2] * costs.gas[2], # gas vom, fuel, and carbon
        history[:, 3] * costs.hydro[2], # hydro vom
        history[:, 4] * costs.phes[3], # phes vom
        )).sum(axis=0)
    Lcoes += costs.phes[4] 
    Lcoes += (CHydro.sum() + CBio.sum())*(costs.hydro[0] + costs.hydro[1] + costs.ac.sum()) # HydroBio fom
    Lcoes /= 1_000_000_000
    Lcoes /= history[:, 0]
    Lcoes += history[:, 1]
    return Lcoes


if __name__ == '__main__':
    fileprinter = Fileprinter(f'Results/Paramsweep{scenario}.csv', 1, [
        'carbon price', 'gas fuel', 'pv capex', 'wind capex', 'LCOE'] + list(range(len(lb))), 
        resume=bool(args.res))
    
    start = not bool(args.res)
    
    hyperparameters = (
        (1.0, 1.5, 0.5, 50, 100, 1), 
        (0.5, 1.0, 0.4, 50, 100, 0.1))
    
    for hp in hyperparameters:
        i=0
        args.ml, args.mu, args.r, disp_step, stag, stag_rate = hp
        for p, carbon_step in enumerate(carbon_price):
            raw_costs.carbon_price = carbon_step
            # ============================================
            # gas fuel cost not being updated properly here??
            # ============================================
            costs = raw_costs.CostFactors()
            for q, gas_step in enumerate(gas_fuel):
                raw_costs.gas[3] = gas_step
                costs = raw_costs.CostFactors()
                for r, pv_step in enumerate(pv_capex):
                    raw_costs.pv[0] = pv_step
                    costs = raw_costs.CostFactors()
                    for s, wind_step in enumerate(wind_capex):
                        raw_costs.onsw[0] = wind_step
                        costs = raw_costs.CostFactors()
                        if q != 3:
                            continue
                        if start:
                            with open(f'Results/History{scenario}.csv', 'w', newline='') as file:
                                writer(file)
                            init = 'latinhypercube'
                            x0=None
                            start=False
                        else:
                            init = select_population(costs)
                            x0 = init[0]
                            if len(init) < args.p:
                                init = 'latinhypercube'
                        i+=1
                        print(i, '/', 108)
                        result, t = Optimise(costs, init, x0, (disp_step, stag, stag_rate))
                        fileprinter([[p,q,r,s,result.fun]+list(result.x)])
    PrintTimekeeper(f'Results/Timekeep-ps-{scenario}.csv')

