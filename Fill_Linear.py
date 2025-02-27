
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
    

    model.t     = pyo.RangeSet(solution.intervals) 

    model.charge =  pyo.Var(model.t, domain=pyo.NonNegativeReals)
    model.discharge=pyo.Var(model.t, domain=pyo.NonNegativeReals)
    model.storage = pyo.Var(model.t, domain=pyo.NonNegativeReals)
    model.hydro =   pyo.Var(model.t, domain=pyo.NonNegativeReals)
    model.bio =     pyo.Var(model.t, domain=pyo.NonNegativeReals)
    model.gas =     pyo.Var(model.t, domain=pyo.NonNegativeReals)
    
    model.constr_charge_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.charge[t] <= solution.GCPHP)
    model.constr_charge_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.charge[t] >= 0)
    model.constr_discharge_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.discharge[t] <= solution.GCPHP)
    model.constr_discharge_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.discharge[t] >= 0)
    model.constr_storage_energy_upper = pyo.Constraint(model.t, rule=lambda m, t: m.storage[t] <= solution.CPHS)
    model.constr_storage_energy_lower = pyo.Constraint(model.t, rule=lambda m, t: m.storage[t] >= 0)
    
    model.constr_hydro_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.hydro[t] <= GCHydro)
    model.constr_hydro_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.hydro[t] >= 0)
    model.constr_bio_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.bio[t] <= GCBio)
    model.constr_bio_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.bio[t] >= 0)
    model.constr_gas_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.gas[t] <= m.GCGas)
    # model.constr_gas_power_upper = pyo.Constraint(model.t, rule=lambda m, t: m.gas[t] <= GCGas)
    model.constr_gas_power_lower = pyo.Constraint(model.t, rule=lambda m, t: m.gas[t] >= 0)
    
    model.constr_max_hydro = pyo.Constraint(rule=lambda m: pyo.summation(m.hydro)*solution.resolution/solution.years <= solution.Hydro_res)
    model.constr_max_bio   = pyo.Constraint(rule=lambda m: pyo.summation(m.bio)*solution.resolution/solution.years <= solution.Bio_res) 

    def constr_state_of_charge(m, t):
        if t==1:
            return m.storage[t] == 0.5 * solution.CPHS - m.discharge[t] * solution.resolution + m.charge[t] * solution.resolution * solution.efficiency
        else:
            return m.storage[t] == m.storage[t-1] - m.discharge[t] * solution.resolution + m.charge[t] * solution.resolution * solution.efficiency
    
    model.constr_storage_state_of_charge = pyo.Constraint(model.t, rule=constr_state_of_charge)
        
    def expr_energy_balance(m, t):
        return (solution.MLoad[t-1].sum()
                + m.charge[t] 
                - solution.MPV[t-1].sum()
                - solution.MOnsW[t-1].sum()
                - solution.CBaseload.sum()
                - m.hydro[t] 
                - m.bio[t] 
                - m.gas[t]
                - m.discharge[t] 
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
    
    x = np.genfromtxt('Results/Optimisation_resultx{}.csv'.format(scenario), delimiter=',', dtype=float)
   
    solution = Solution(x)
    solution.CGas = 2/5*np.ones(5)
    model = Fill_linear(solution)
    
    GHydro = np.array([model.hydro[i].value for i in model.hydro])
    GBio   = np.array([model.bio[i].value   for i in model.bio])
    GGas   = np.array([model.gas[i].value   for i in model.gas])

    deficit = Reliability(solution, GHydro+GBio+GGas)
