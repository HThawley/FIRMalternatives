
"""
This module is not used in the optimisation. 
It is used to verify that Fill.py works as intended 
    i.e. provides a good approximation of a perfect Fill
"""

from datetime import datetime as dt
import numpy as np
import pyomo.environ as pyo

def Fill_linear(solution):
    
    print("Instantiating optimiser:", dt.now())
    model = pyo.ConcreteModel()
        
    GCHydro = solution.CHydro.sum()
    GCBio = solution.CBio.sum()
    model.GCGas = pyo.Var(domain=pyo.NonNegativeReals)
    # GCGas = solution.CGas.sum()

    model.t = pyo.RangeSet(solution.intervals) 
    model.n = pyo.RangeSet(solution.nodes) 

    model.charge = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MCharge[t-1, n-1],
        )
    model.discharge = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MDischarge[t-1, n-1],
        )
    model.storage = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MStorage[t-1, n-1],
        )
    model.hydro = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MHydro[t-1, n-1],
        )
    model.bio = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MBio[t-1, n-1],
        )
    model.gas = pyo.Var(
        model.t, 
        model.n, 
        domain=pyo.NonNegativeReals, 
        initialize = lambda m, t, n: solution.MGas[t-1, n-1],
        )
    
    model.constr_charge_power_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.charge[t, n] <= solution.CPHP[n-1])
    model.constr_charge_power_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.charge[t, n] >= 0)
    model.constr_discharge_power_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.discharge[t, n] <= solution.CPHP[n-1])
    model.constr_discharge_power_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.discharge[t, n] >= 0)
    model.constr_storage_energy_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.storage[t, n] <= solution.CPHS[n-1])
    model.constr_storage_energy_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.storage[t, n] >= 0)
    
    model.constr_hydro_power_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.hydro[t, n] <= solution.CHydro[n-1])
    model.constr_hydro_power_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.hydro[t] >= 0)
    model.constr_bio_power_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.bio[t, n] <= solution.CBio[n-1])
    model.constr_bio_power_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.bio[t, n] >= 0)
    model.constr_gas_power_upper = pyo.Constraint(model.t, rule=lambda m, t, n: m.gas[t, n] <= m.CGas[n-1])
    model.constr_gas_power_lower = pyo.Constraint(model.t, rule=lambda m, t, n: m.gas[t, n] >= 0)
    
    model.constr_max_hydro = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*solution.resolution/solution.years <= solution.Hydro_res)
    model.constr_max_bio   = pyo.Constraint(rule=lambda m: pyo.summation(m.bio)*solution.resolution/solution.years <= solution.Bio_res) 

    def constr_state_of_charge(m, t, n):
        if t==1:
            return m.storage[t, n] == 0.5 * solution.CPHS[n-1] - m.discharge[t, n] * solution.resolution + m.charge[t, n] * solution.resolution * solution.efficiency
        else:
            return m.storage[t, n] == m.storage[t-1, n] - m.discharge[t, n] * solution.resolution + m.charge[t, n] * solution.resolution * solution.efficiency
    
    model.constr_storage_state_of_charge = pyo.Constraint(model.t, model.n, rule=constr_state_of_charge)
        
    def expr_energy_balance(m, t, n):
        return (solution.MLoad[t-1, n]
                + m.charge[t, n] 
                - solution.MPV[t-1, n]
                - solution.MOnsW[t-1, n]
                - solution.CBaseload[n-1]
                - m.hydro[t, n]
                - m.bio[t, n] 
                - m.gas[t, n]
                - m.discharge[t, n] 
                )
    
    model.energy_balance = pyo.Expression(model.t, rule=expr_energy_balance)
    model.constr_energy_balance = pyo.Constraint(model.t, rule=lambda m, t : m.energy_balance[t]<=0)
    
    model.obj = pyo.Objective(rule=lambda m: m.GCGas*10000 + pyo.summation(m.gas))
    # model.obj = pyo.Objective(rule=lambda m: pyo.summation(m.gas))
        
    start=dt.now()
    print("Optimising. Start:",start)
    optimiser = pyo.SolverFactory('gurobi')
    optimiser.solve(model)
    end=dt.now()
    print("Optimisation took:", end-start)
    
    
    return model

if __name__ == '__main__':
    from Input import * 
    from Simulation import Reliability
    
    costs = Raw_Costs(scenario, DClengths, undersea_mask, network_mask).CostFactors()
    
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
   
    solution = Solution(x)
    solution._evaluate(costs)
    
    solution.CGas = 2/5*np.ones(5)
    model = Fill_linear(solution)
    
    
    # This is likely now broken \/
    GHydro = np.array([model.hydro[i].value for i in model.hydro])
    GBio   = np.array([model.bio[i].value   for i in model.bio])
    GGas   = np.array([model.gas[i].value   for i in model.gas])

    deficit = Reliability(solution, GHydro+GBio+GGas)
