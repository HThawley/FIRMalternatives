import numpy as np 
import pandas as pd
from numba import njit
from csv import writer


from Input import * 
from Costs import Raw_Costs 
from Optimisation import Optimise
# from Timekeeper import keeptime, PrintTimekeeper, timekeeper


## Parameters to sweep 
raw_costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask)

pv_capex = raw_costs.pv[0]
pv_capex = 0.75*pv_capex, pv_capex, 1.25*pv_capex

wind_capex = raw_costs.onsw[0]
wind_capex = 0.75*wind_capex, wind_capex, 1.25*wind_capex

gas_fuel = raw_costs.gas[3]
gas_fuel = 0.75*gas_fuel, gas_fuel, 1.25*gas_fuel

carbon_price = (0, 35, 70, 140)

costs = raw_costs.CostFactors()

@keeptime
def select_population(costs):
    history = pd.read_csv(f'Results/History{scenario}.csv', header=None, 
                          usecols=[0,1,2,3,4]+list(range(7, 14+len(lb))))
    history = history.drop_duplicates(subset=range(12, history.shape[1])).to_numpy()
    Lcoes = calculate_costs(history, costs)
    sort_array = np.argsort(Lcoes)
    noptimaln = min(len(Lcoes), args.p)
    history = history[sort_array[:noptimaln], 12:]

    return history

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

@keeptime
@njit
def calculate_costs(history, costs):
    Lcoes = np.stack((
        (history[:,5:12] * costs.hvdc.sum(axis=0)).sum(axis=1), # hvdc capex and fom 
        history[:, 12:      12+pidx].sum(axis=1) * (costs.pv[0]   + costs.pv[1]   + costs.ac.sum()), # pv capex and fom
        history[:, 12+pidx: 12+widx].sum(axis=1) * (costs.onsw[0] + costs.onsw[1] + costs.ac.sum()), # wind capex and fom
        history[:, 12+widx: 12+gidx].sum(axis=1) * (costs.gas[0]  + costs.gas[1]  + costs.ac.sum()), # gas capex and fom
        history[:, 12+gidx: 12+sidx].sum(axis=1) * (costs.phes[0] + costs.phes[2] ), # phes capex (power) and fom
        history[:, 12+sidx] * costs.phes[1], # phes capex (energy)
        
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
    
    start = True
    args.ml = 0.5
    args.mu = 1.5
    args.r = 0.4
    for s, carbon_step in enumerate(carbon_price):
        raw_costs.UpdateCarbonPrice(carbon_step)
        costs = raw_costs.CostFactors()
        for r, gas_step in enumerate(gas_fuel):
            raw_costs.gas[3] = gas_step
            costs = raw_costs.CostFactors()
            for p, pv_step in enumerate(pv_capex):
                raw_costs.pv[0] = pv_step
                costs = raw_costs.CostFactors()
                for q, wind_step in enumerate(wind_capex):
                    raw_costs.onsw[0] = wind_step
                    costs = raw_costs.CostFactors()
            
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
                    result, t = Optimise(costs, init, x0, (25, 50, 1))
                    
                    with open(f'Results/Opt_result{scenario}-{p}-{q}-{r}-{s}.csv', 'w', newline='') as csvfile:
                        writer(csvfile).writerow([result.fun] + list(result.x))

    args.ml = 0.25
    args.mu = 0.5
    args.r = 0.15
    for s, carbon_step in enumerate(carbon_price):
        raw_costs.UpdateCarbonPrice(carbon_step)
        costs = raw_costs.CostFactors()
        for r, gas_step in enumerate(gas_fuel):
            raw_costs.gas[3] = gas_step
            costs = raw_costs.CostFactors()
            for p, pv_step in enumerate(pv_capex):
                raw_costs.pv[0] = pv_step
                costs = raw_costs.CostFactors()
                for q, wind_step in enumerate(wind_capex):
                    raw_costs.onsw[0] = wind_step
                    costs = raw_costs.CostFactors()
                    
                    init = select_population(costs)
                    x0 = init[0]
                    if len(init) < args.p:
                        init = 'latinhypercube'
                    result, t = Optimise(costs, init, x0, (50, 100, 1e-6))

                    with open(f'Results/Opt_result{scenario}-{p}-{q}-{r}-{s}.csv', 'w', newline='') as csvfile:
                        writer(csvfile).writerow([result.fun] + list(result.x))

